import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("credit_risk_model.joblib")

st.title("Credit Risk Assessment Dashboard")
st.write("Enter applicant details to predict risk category (Low / Moderate / High).")

# Sidebar inputs
annual_income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
debt_to_income = st.sidebar.number_input("Debt-to-Income Ratio", min_value=0.0, value=20.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=15000)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
term = st.sidebar.selectbox("Loan Term (months)", [36, 60])
emp_length = st.sidebar.number_input("Employment Length (years)", min_value=0, value=5)
inquiries_last_12m = st.sidebar.number_input("Credit Inquiries (12 months)", min_value=0, value=1)
months_since_last_credit_inquiry = st.sidebar.number_input("Months Since Last Credit Inquiry", min_value=0, value=6)
account_never_delinq_percent = st.sidebar.number_input("Account Never Delinquent (%)", min_value=0, value=95)

homeownership = st.sidebar.selectbox("Homeownership", ["OWN","RENT","MORTGAGE"])
verified_income = st.sidebar.selectbox("Verified Income", ["Verified","Not Verified","Source Verified"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["debt_consolidation","medical","small_business","other"])
application_type = st.sidebar.selectbox("Application Type", ["individual","joint"])
state = st.sidebar.text_input("State (e.g., NY)", "NY")

# Predict button
if st.sidebar.button("Predict Risk Category"):
    new_applicant = pd.DataFrame([{
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "term": term,
        "delinq_2y": 0,
        "months_since_last_delinq": -1,
        "num_historical_failed_to_pay": 0,
        "public_record_bankrupt": 0,
        "tax_liens": 0,
        "emp_length": emp_length,
        "inquiries_last_12m": inquiries_last_12m,
        "months_since_last_credit_inquiry": months_since_last_credit_inquiry,
        "num_accounts_120d_past_due": 0,
        "account_never_delinq_percent": account_never_delinq_percent,
        "homeownership": homeownership,
        "verified_income": verified_income,
        "loan_purpose": loan_purpose,
        "application_type": application_type,
        "state": state
    }])

    prediction = model.predict(new_applicant)[0]
    probs = model.predict_proba(new_applicant)[0]
    classes = model.classes_

    st.subheader("Predicted Risk Category")
    st.write(prediction)

    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({"Category": classes, "Probability": probs})
    st.bar_chart(prob_df.set_index("Category"))
