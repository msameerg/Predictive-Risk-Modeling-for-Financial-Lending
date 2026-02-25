import streamlit as st
import joblib
import pandas as pd

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Credit Risk AI", layout="wide")
st.title("Financial Risk Assessment System")
st.markdown("---")

# 2. LOAD THE REFINED MODEL
try:
    model = joblib.load('credit_risk_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'credit_risk_model.pkl' is in the directory.")
    st.stop()

# 3. INPUT SECTION
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
    # Ensure dataframe matches the features used in your refined_pipeline
    input_df = pd.DataFrame([{
        'loan_amount': loan_amount,
        'property_value': property_value,
        'income': income,
        'Credit_Score': credit_score,
        'dtir1': dti,
        'loan_type': loan_type,
        'loan_limit': loan_limit,
        'approv_in_adv': approv_in_adv,
        'Credit_Worthiness': credit_worthiness
    }])

    # Get Probability
    probability = model.predict_proba(input_df)[0][1]
    
    # 5. RESULT DISPLAY
    st.markdown("### Risk Analysis Result")
    
    if probability < 0.30:
        st.success(f"**LOW RISK:** Default Probability is {probability:.2%}")
        st.info("Recommendation: Proceed with standard approval process.")
    elif probability < 0.60:
        st.warning(f"**MODERATE RISK:** Default Probability is {probability:.2%}")
        st.info("Recommendation: Request additional documentation.")
    else:
        st.error(f"**HIGH RISK:** Default Probability is {probability:.2%}")
        st.info("Recommendation: High potential for default. Manual review required.")
