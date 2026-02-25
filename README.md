# Predictive-Risk-Modeling-for-Financial-Lending

##Click here to open the Live App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-risk-modeling-for-financial-lending.streamlit.app/)
## Overview

This project develops a machine learning system to predict the probability of loan default. It transforms raw financial data into a risk-scoring mechanism used to support lending decisions in banking and fintech environments.

## Technical Problem: Data Leakage

The initial model achieved 100% accuracy, indicating significant data leakage. Analysis identified that features like rate_of_interest and upfront_charges were updated post-loan status. To ensure real-world applicability, these features were removed to focus on pre-approval variables.

## Model Iteration: Feature Dominance

A second iteration revealed that the feature credit_type_EQUI accounted for over 70% of model importance. This administrative category acted as a "shortcut" rather than a predictive financial indicator. The final model was refined by removing this feature, forcing the algorithm to learn from actual borrower health metrics such as Debt-to-Income (DTI) ratio and property value.

## Implementation Details

* **Architecture:** Scikit-Learn Pipeline for end-to-end processing.
* **Preprocessing:** Median imputation for missing values and One-Hot Encoding for categorical variables.
* **Imbalance Handling:** Utilized scale_pos_weight to manage the minority class of defaulters without synthetic data generation.
* **Algorithms Benchmarked:** Logistic Regression, Random Forest, and XGBoost.

## Results

XGBoost was selected as the production model due to superior performance in ranking and risk detection.

| Metric | Result |
| --- | --- |
| ROC-AUC | 0.89 |
| Recall (Default Class) | 70% |
| Precision (Default Class) | 73% |

## Business Impact

By adjusting the classification threshold to 0.3, the model captures 70% of potential defaulters. This proactive risk identification allows lenders to minimize capital loss while maintaining a high precision rate for safe borrowers.

## Key Risk Drivers

The model identified the following as the primary predictors of default:

1. Property Value
2. Debt-to-Income Ratio (dtir1)
3. Credit Worthiness
4. Loan Limit

## Deployment

The final pipeline is serialized into a .pkl file, enabling immediate integration into a Streamlit web interface for real-time risk assessment.
