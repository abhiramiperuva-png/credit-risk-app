"""
CREDIT RISK DASHBOARD (Streamlit)
==================================
Interactive dashboard for the loan risk classification project.

Run with:
    pip install streamlit scikit-learn pandas numpy joblib plotly
    streamlit run app.py

Expects `loans_full_schema.csv` in the same folder. If `credit_risk_model.joblib`
already exists (saved by your training script) it will be loaded; otherwise the
model is trained on the fly (cached, so it only happens once per session).
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

RANDOM_STATE = 42
DATA_PATH = "loans_full_schema.csv"
MODEL_PATH = "credit_risk_model.joblib"

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------------
# SHARED PIPELINE CODE (same logic as the training script)
# ---------------------------------------------------------------------------
LOW_RISK_STATUSES = ["Fully Paid", "Current"]
MODERATE_RISK_STATUSES = ["In Grace Period", "Late (16-30 days)"]
HIGH_RISK_STATUSES = ["Late (31-120 days)", "Charged Off", "Default"]

REGION_MAP = {
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
    'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
    'ND': 'Midwest', 'SD': 'Midwest',
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South', 'SC': 'South',
    'VA': 'South', 'DC': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South', 'MS': 'South',
    'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South',
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West',
    'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West', 'HI': 'West', 'OR': 'West', 'WA': 'West'
}

RAW_NUMERIC = [
    'annual_income', 'debt_to_income', 'loan_amount', 'interest_rate', 'term',
    'delinq_2y', 'months_since_last_delinq', 'num_historical_failed_to_pay',
    'public_record_bankrupt', 'tax_liens', 'emp_length', 'inquiries_last_12m',
    'months_since_last_credit_inquiry', 'num_accounts_120d_past_due',
    'account_never_delinq_percent',
]
RAW_CATEGORICAL = ['homeownership', 'verified_income', 'loan_purpose', 'application_type']
RAW_STATE = ['state']


def map_loan_status_to_risk(status: str) -> str:
    status = str(status).strip()
    if status in LOW_RISK_STATUSES:
        return "Low Risk"
    if status in MODERATE_RISK_STATUSES:
        return "Moderate Risk"
    if status in HIGH_RISK_STATUSES:
        return "High Risk"
    return "Unknown"


class RiskFeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, lower_pct=0.01, upper_pct=0.99):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = X.copy()
        X["region"] = X["state"].map(REGION_MAP)
        self.dti_median_ = X["debt_to_income"].median()
        self.emp_length_median_ = X["emp_length"].median()
        self.credit_inquiry_median_ = X["months_since_last_credit_inquiry"].median()
        self.income_bounds_ = (
            X["annual_income"].quantile(self.lower_pct),
            X["annual_income"].quantile(self.upper_pct),
        )
        self.dti_bounds_ = (
            X["debt_to_income"].quantile(self.lower_pct),
            X["debt_to_income"].quantile(self.upper_pct),
        )
        self.region_rate_means_ = X.groupby("region")["interest_rate"].mean()
        self.region_rate_global_mean_ = X["interest_rate"].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X["region"] = X["state"].map(REGION_MAP)
        X["debt_to_income"] = X["debt_to_income"].fillna(self.dti_median_)
        X["months_since_last_delinq"] = X["months_since_last_delinq"].fillna(-1)
        X["months_since_last_credit_inquiry"] = X["months_since_last_credit_inquiry"].fillna(
            self.credit_inquiry_median_
        )
        X["emp_length"] = X["emp_length"].fillna(self.emp_length_median_)
        X["num_accounts_120d_past_due"] = X["num_accounts_120d_past_due"].fillna(0)
        low, high = self.income_bounds_
        X["annual_income"] = X["annual_income"].clip(lower=low, upper=high)
        low, high = self.dti_bounds_
        X["debt_to_income"] = X["debt_to_income"].clip(lower=low, upper=high)
        X["income_to_loan_ratio"] = (X["annual_income"] / X["loan_amount"].replace(0, np.nan)).fillna(0)
        X["region_avg_interest_rate"] = (
            X["region"].map(self.region_rate_means_).fillna(self.region_rate_global_mean_)
        )
        return X.drop(columns=["state"])


NUMERIC_FEATURES = RAW_NUMERIC + ["income_to_loan_ratio", "region_avg_interest_rate"]
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ["region"]


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.drop_duplicates()
    df["risk_category"] = df["loan_status"].apply(map_loan_status_to_risk)
    df = df[df["risk_category"] != "Unknown"]
    return df


@st.cache_resource(show_spinner=True)
def get_model(df, use_saved=True, tune=False):
    X = df[RAW_NUMERIC + RAW_CATEGORICAL + RAW_STATE].copy()
    y = df["risk_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    if use_saved and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
        ])
        rf_pipeline = Pipeline(steps=[
            ("clean", RiskFeatureCleaner()),
            ("preprocess", preprocessor),
            ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ])
        if tune:
            param_grid = {
                "rf__n_estimators": [200, 400],
                "rf__max_depth": [None, 10, 20],
                "rf__min_samples_leaf": [1, 2, 5],
                "rf__class_weight": ["balanced", "balanced_subsample"],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            search = GridSearchCV(rf_pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
        else:
            rf_pipeline.set_params(rf__n_estimators=300, rf__class_weight="balanced")
            rf_pipeline.fit(X_train, y_train)
            model = rf_pipeline
        joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred


# ---------------------------------------------------------------------------
# LOAD DATA + MODEL
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Couldn't find `{DATA_PATH}` in this folder. Place your CSV next to app.py and rerun.")
    st.stop()

with st.spinner("Loading data..."):
    df = load_data()

st.sidebar.title("⚙️ Controls")
retrain = st.sidebar.checkbox("Force retrain (ignore saved model)", value=False)
tune = st.sidebar.checkbox("Run hyperparameter tuning (slower)", value=False)

with st.spinner("Loading / training model..."):
    model, X_train, X_test, y_train, y_test, y_pred = get_model(df, use_saved=not retrain, tune=tune)

labels = ["Low Risk", "Moderate Risk", "High Risk"]

# ---------------------------------------------------------------------------
# HEADER + TOP KPIs
# ---------------------------------------------------------------------------
st.title("📊 Credit Risk Assessment Dashboard")
st.caption("Peer-to-peer loan portfolio — risk classification at origination")

macro_f1 = f1_score(y_test, y_pred, average="macro")
high_risk_mask_true = (y_test == "High Risk")
high_risk_mask_pred = (y_pred == "High Risk")
caught = int((high_risk_mask_true & high_risk_mask_pred).sum())
total_high_risk = int(high_risk_mask_true.sum())
recall_high_risk = caught / total_high_risk if total_high_risk else float("nan")
loan_volume_at_risk = X_test.loc[high_risk_mask_true, "loan_amount"].sum()
loan_volume_flagged = X_test.loc[high_risk_mask_true & high_risk_mask_pred, "loan_amount"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total loans (dataset)", f"{len(df):,}")
k2.metric("Model Macro F1", f"{macro_f1:.3f}")
k3.metric("High-Risk Recall", f"{recall_high_risk:.1%}", help="% of truly high-risk test loans the model caught")
k4.metric("At-Risk $ Flagged", f"${loan_volume_flagged:,.0f}", f"of ${loan_volume_at_risk:,.0f} total")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏦 Portfolio Overview", "🧠 Model Performance", "🔍 Feature Drivers", "🧮 Score a New Applicant"]
)

# ---------------------------------------------------------------------------
# TAB 1 — PORTFOLIO OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        risk_counts = df["risk_category"].value_counts().reindex(labels)
        fig = px.pie(
            names=risk_counts.index, values=risk_counts.values,
            title="Risk Category Distribution",
            color=risk_counts.index,
            color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"},
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        region_df = df.copy()
        region_df["region"] = region_df["state"].map(REGION_MAP)
        region_risk = region_df.groupby(["region", "risk_category"]).size().reset_index(name="count")
        fig2 = px.bar(
            region_risk, x="region", y="count", color="risk_category",
            title="Risk by Region", barmode="stack",
            color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Hypothesis checks")
    hc1, hc2, hc3 = st.columns(3)

    with hc1:
        fig3 = px.box(df, x="risk_category", y="debt_to_income", category_orders={"risk_category": labels},
                       title="H1: DTI vs Risk", color="risk_category",
                       color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)

    with hc2:
        verif = df.groupby(["verified_income", "risk_category"]).size().reset_index(name="count")
        verif["pct"] = verif.groupby("verified_income")["count"].transform(lambda x: x / x.sum())
        fig4 = px.bar(verif, x="verified_income", y="pct", color="risk_category", barmode="stack",
                       title="H2: Income Verification vs Risk (%)",
                       color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"})
        st.plotly_chart(fig4, use_container_width=True)

    with hc3:
        own = df.groupby(["homeownership", "risk_category"]).size().reset_index(name="count")
        own["pct"] = own.groupby("homeownership")["count"].transform(lambda x: x / x.sum())
        fig5 = px.bar(own, x="homeownership", y="pct", color="risk_category", barmode="stack",
                       title="H3: Homeownership vs Risk (%)",
                       color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"})
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Filter the portfolio")
    f1, f2, f3 = st.columns(3)
    sel_purpose = f1.multiselect("Loan purpose", sorted(df["loan_purpose"].unique()))
    sel_home = f2.multiselect("Homeownership", sorted(df["homeownership"].unique()))
    sel_risk = f3.multiselect("Risk category", labels)

    filtered = df.copy()
    if sel_purpose:
        filtered = filtered[filtered["loan_purpose"].isin(sel_purpose)]
    if sel_home:
        filtered = filtered[filtered["homeownership"].isin(sel_home)]
    if sel_risk:
        filtered = filtered[filtered["risk_category"].isin(sel_risk)]

    st.dataframe(
        filtered[["state", "homeownership", "verified_income", "loan_purpose",
                  "annual_income", "debt_to_income", "loan_amount", "interest_rate",
                  "risk_category"]].head(200),
        use_container_width=True,
    )
    st.caption(f"Showing {min(len(filtered), 200):,} of {len(filtered):,} matching loans")

# ---------------------------------------------------------------------------
# TAB 2 — MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig_cm = px.imshow(
        cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Business impact")
    st.markdown(f"""
- Of **{total_high_risk}** genuinely high-risk loans in the test set, the model correctly flagged **{caught}** (**{recall_high_risk:.1%} recall**).
- Total loan volume in the truly high-risk group: **${loan_volume_at_risk:,.0f}**
- Loan volume the model would flag for manual review: **${loan_volume_flagged:,.0f}**
""")

# ---------------------------------------------------------------------------
# TAB 3 — FEATURE DRIVERS
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("What drives the model's risk predictions")
    try:
        ohe = model.named_steps["preprocess"].named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
        all_feature_names = np.array(NUMERIC_FEATURES + list(cat_feature_names))
        importances = model.named_steps["rf"].feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]

        imp_df = pd.DataFrame({
            "feature": all_feature_names[top_idx],
            "importance": importances[top_idx],
        }).sort_values("importance")

        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                          title="Top 15 Feature Importances", color="importance",
                          color_continuous_scale="Viridis")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not extract feature importances from this model: {e}")

# ---------------------------------------------------------------------------
# TAB 4 — SCORE A NEW APPLICANT
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Try the model on a hypothetical applicant")
    c1, c2, c3 = st.columns(3)

    with c1:
        annual_income = st.number_input("Annual income ($)", 0, 2_000_000, 65000, step=1000)
        loan_amount = st.number_input("Loan amount ($)", 500, 100_000, 15000, step=500)
        interest_rate = st.number_input("Interest rate (%)", 0.0, 40.0, 12.0, step=0.1)
        term = st.selectbox("Term (months)", [36, 60])
        debt_to_income = st.number_input("Debt-to-income (%)", 0.0, 100.0, 20.0, step=0.5)

    with c2:
        emp_length = st.number_input("Employment length (years)", 0, 40, 5)
        homeownership = st.selectbox("Homeownership", sorted(df["homeownership"].unique()))
        verified_income = st.selectbox("Income verification", sorted(df["verified_income"].unique()))
        loan_purpose = st.selectbox("Loan purpose", sorted(df["loan_purpose"].unique()))
        application_type = st.selectbox("Application type", sorted(df["application_type"].unique()))

    with c3:
        state = st.selectbox("State", sorted(REGION_MAP.keys()))
        delinq_2y = st.number_input("Delinquencies (last 2y)", 0, 20, 0)
        inquiries_last_12m = st.number_input("Credit inquiries (last 12m)", 0, 30, 1)
        public_record_bankrupt = st.number_input("Public record bankruptcies", 0, 5, 0)
        account_never_delinq_percent = st.slider("% accounts never delinquent", 0.0, 100.0, 90.0)

    if st.button("Predict risk category", type="primary"):
        applicant = pd.DataFrame([{
            "annual_income": annual_income, "debt_to_income": debt_to_income,
            "loan_amount": loan_amount, "interest_rate": interest_rate, "term": term,
            "delinq_2y": delinq_2y, "months_since_last_delinq": np.nan,
            "num_historical_failed_to_pay": 0, "public_record_bankrupt": public_record_bankrupt,
            "tax_liens": 0, "emp_length": emp_length, "inquiries_last_12m": inquiries_last_12m,
            "months_since_last_credit_inquiry": 6, "num_accounts_120d_past_due": 0,
            "account_never_delinq_percent": account_never_delinq_percent,
            "homeownership": homeownership, "verified_income": verified_income,
            "loan_purpose": loan_purpose, "application_type": application_type, "state": state,
        }])
        pred = model.predict(applicant)[0]
        proba = model.predict_proba(applicant)[0]
        proba_df = pd.DataFrame({"Risk": model.classes_, "Probability": proba}).sort_values("Probability", ascending=False)

        color = {"Low Risk": "🟢", "Moderate Risk": "🟠", "High Risk": "🔴"}[pred]
        st.success(f"### {color} Predicted category: **{pred}**")
        fig_p = px.bar(proba_df, x="Risk", y="Probability", color="Risk",
                        color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"})
        st.plotly_chart(fig_p, use_container_width=True)
