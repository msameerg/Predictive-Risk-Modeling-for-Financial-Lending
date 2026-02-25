import streamlit as st
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
