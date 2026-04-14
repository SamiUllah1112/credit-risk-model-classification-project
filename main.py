import streamlit as st
from prediction_helpher import predict
st.title("Lauki Finance: Credit Risk Modeling")

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(1)

# Row 1 - Basic Info
with row1[0]:
    age = st.number_input('Age', min_value=18, max_value=100, step=1)

with row1[1]:
    income = st.number_input('Income', min_value=0, value=1200000)

with row1[2]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2550000)

# Row 2 - Loan Details
with row2[0]:
    loan_tenure_months = st.number_input('Loan Tenure (Months)', min_value=0, step=1)

with row2[1]:
    net_disbursement = st.number_input('Net Disbursement', min_value=0)

with row2[2]:
    number_of_open_accounts = st.number_input('Number of Open Accounts', min_value=0, step=1)

# Row 3 - Credit Info
with row3[0]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, max_value=100.0, step=1.0)

with row3[1]:
    delinquent_ratio = st.number_input('Delinquent Ratio', min_value=0.0, max_value=100.0, step=1.0)

with row3[2]:
    avg_dpd_per_delinquent = st.number_input('Avg DPD per Delinquent', min_value=0.0, max_value=100.0, step=1.0)

# Row 4 - Categorical + Loan to Income display
with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented'])

with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Personal'])

with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Secured', 'Unsecured'])

# Derived Feature — displayed always, outside button
loan_to_income = loan_amount / income if income > 0 else 0
st.text(f"Loan to Income Ratio: {loan_to_income:.2f}")  # ← always visible

# Row 5 - Button
with row5[0]:
    if st.button('Calculate Risk'):
        probability, credit_score, rating = predict(age, income, loan_amount, net_disbursement, loan_tenure_months,
       number_of_open_accounts, credit_utilization_ratio,
       delinquent_ratio, avg_dpd_per_delinquent, residence_type,
        loan_purpose,loan_type)

        
        st.write(f"default probability: {probability:.2%}")
        st.write(f"credit score: {credit_score}")
        st.write(f"rating: {rating}")

