import streamlit as st
import joblib
import pandas as pd

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Credit Risk AI", layout="wide")
st.title("🛡️ Financial Risk Assessment System")
st.markdown("---")

# 2. LOAD THE REFINED MODEL
# Ensure this file is in the same folder as app.py
try:
    model = joblib.load('credit_risk_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'credit_risk_model.pkl' is in the directory.")
    st.stop()

# 3. INPUT SECTION - Organized into Columns
st.subheader("Customer Financial Profile")
col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=50000)
    property_value = st.number_input("Property Value ($)", min_value=0, value=150000)
    income = st.number_input("Monthly Income ($)", min_value=0, value=4500)

with col2:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 1.0, 0.35)
    loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"])

with col3:
    loan_limit = st.selectbox("Loan Limit", ["cf", "ncf"])
    approv_in_adv = st.selectbox("Approved in Advance", ["nopre", "pre"])
    credit_worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])

# 4. PREDICTION LOGIC
if st.button("Generate Risk Report"):
    # CREATE DATAFRAME: Columns must match the X_refined structure exactly
    input_df = pd.DataFrame([{
        'loan_amount': loan_amount,
        'property_value': property_value,
        'income': income,
        'Credit_Score': credit_score,
        'dtir1': dti,
        'loan_type': loan_type,
        'loan_limit': loan_limit,
        'approv_in_adv': approv_in_adv,
        'Credit_Worthiness': credit_worthiness,
        # Add any other missing columns from your X_refined with default/mean values
    }])

    # Probability Prediction
    probability = model.predict_proba(input_df)[0][1]
    
    # 5. RESULT DISPLAY
    st.markdown("### Risk Analysis Result")
    
    if probability < 0.30:
        st.success(f"**LOW RISK:** Default Probability is {probability:.2%}")
        st.info("Recommendation: Proceed with standard approval process.")
    elif probability < 0.60:
        st.warning(f"**MODERATE RISK:** Default Probability is {probability:.2%}")
        st.info("Recommendation: Request additional documentation or higher down payment.")
    else:
        st.error(f"**HIGH RISK:** Default Probability is {probability:.2%}")
        st.info("Recommendation: High potential for default. Manual review required.")import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. SET UP THE PAGE
st.set_page_config(page_title="Credit Risk AI", layout="centered")
st.title("🛡️ Financial Risk Predictor")
st.markdown("Enter customer details to calculate the probability of loan default.")

# 2. LOAD THE REFINED MODEL
# Make sure you saved your REFINED model pkl as 'credit_risk_model.pkl'
model = joblib.load('credit_risk_model.pkl')

# 3. CREATE INPUTS (We'll use the top features we identified)
with st.container():
    st.subheader("Customer Financial Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        property_value = st.number_input("Property Value ($)", min_value=0, value=100000)
        income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=50000)
    
    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"])

# 4. PREDICTION LOGIC
if st.button("Calculate Risk Score"):
    # We create a dataframe with the exact columns our model expects
    # For simplicity in this demo, we'll fill other columns with defaults/medians
    # In a full app, you'd map every input carefully
    input_data = pd.DataFrame([{
        'property_value': property_value,
        'income': income,
        'loan_amount': loan_amount,
        'Credit_Score': credit_score,
        'dtir1': dti,
        'loan_type': loan_type,
        # ... add other necessary columns with dummy/default values
    }])
    
    # Get probability
    prob = model.predict_proba(input_data)[0][1]
    
    # 5. DISPLAY RESULTS
    st.divider()
    if prob < 0.3:
        st.success(f"**LOW RISK:** Default Probability is {prob:.2%}")
    elif prob < 0.6:
        st.warning(f"**MEDIUM RISK:** Default Probability is {prob:.2%}")
    else:
        st.error(f"**HIGH RISK:** Default Probability is {prob:.2%}")
