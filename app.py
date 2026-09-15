"""
app.py
Streamlit web app: Loan Default Prediction using Logistic Regression.
Run with:  streamlit run app.py
"""

import pandas as pd
import joblib
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")

# ------------------------------------------------------------------
# Load trained model + feature column order
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_default_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_artifacts()

# ------------------------------------------------------------------
# Min-max ranges used to scale raw numeric inputs
# (must match the ranges the training data was originally scaled with)
# ------------------------------------------------------------------
RANGES = {
    "age": (18, 69),
    "income": (15000, 149999),
    "loanamount": (5000, 249999),
    "creditscore": (300, 849),
    "monthsemployed": (0, 119),
    "numcreditlines": (1, 4),
    "interestrate": (2.0, 25.0),
    "dtiratio": (0.1, 0.9),
}
LOAN_TERMS = [12, 24, 36, 48, 60]  # min=12, max=60 for scaling


def scale(value, col):
    lo, hi = RANGES[col]
    return (value - lo) / (hi - lo)


def scale_loan_term(term):
    return (term - min(LOAN_TERMS)) / (max(LOAN_TERMS) - min(LOAN_TERMS))


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("💰 Loan Default Prediction")
st.write(
    "Fill in the applicant's details below. The model (Logistic Regression) "
    "will predict whether the loan is likely to **default**."
)

st.header("👤 Personal Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=69, value=35)
    income = st.number_input("Annual Income ($)", min_value=15000, max_value=149999, value=50000, step=1000)
    education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
with col2:
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    months_employed = st.number_input("Months Employed", min_value=0, max_value=119, value=24)

st.header("🏦 Loan Details")
col3, col4 = st.columns(2)
with col3:
    loan_amount = st.number_input("Loan Amount ($)", min_value=5000, max_value=249999, value=20000, step=500)
    interest_rate = st.number_input("Interest Rate (%)", min_value=2.0, max_value=25.0, value=10.0, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", LOAN_TERMS, index=2)
with col4:
    dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.1, max_value=0.9, value=0.3, step=0.01)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=1, max_value=4, value=2)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=849, value=650)

st.header("📌 Other Factors")
col5, col6, col7 = st.columns(3)
with col5:
    has_mortgage = st.radio("Has Mortgage?", ["Yes", "No"], horizontal=True)
with col6:
    has_dependents = st.radio("Has Dependents?", ["Yes", "No"], horizontal=True)
with col7:
    has_cosigner = st.radio("Has Co-signer?", ["Yes", "No"], horizontal=True)

loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

st.markdown("---")

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("🔍 Predict Default Risk", use_container_width=True):

    # Build a dict with all feature columns initialised to 0
    input_data = {col: 0 for col in feature_columns}

    # Scaled numeric features
    input_data["age"] = scale(age, "age")
    input_data["income"] = scale(income, "income")
    input_data["loanamount"] = scale(loan_amount, "loanamount")
    input_data["creditscore"] = scale(credit_score, "creditscore")
    input_data["monthsemployed"] = scale(months_employed, "monthsemployed")
    input_data["numcreditlines"] = scale(num_credit_lines, "numcreditlines")
    input_data["interestrate"] = scale(interest_rate, "interestrate")
    input_data["loanterm"] = scale_loan_term(loan_term)
    input_data["dtiratio"] = scale(dti_ratio, "dtiratio")

    # Binary features
    input_data["hasmortgage"] = 1 if has_mortgage == "Yes" else 0
    input_data["hasdependents"] = 1 if has_dependents == "Yes" else 0
    input_data["hascosigner"] = 1 if has_cosigner == "Yes" else 0

    # One-hot: education
    edu_key = f"education_{education.lower().replace(' ', '_')}"
    if edu_key in input_data:
        input_data[edu_key] = 1

    # One-hot: employment type
    emp_key = f"employmenttype_{employment_type.lower()}"
    if emp_key in input_data:
        input_data[emp_key] = 1

    # One-hot: marital status
    mar_key = f"maritalstatus_{marital_status.lower()}"
    if mar_key in input_data:
        input_data[mar_key] = 1

    # One-hot: loan purpose
    purpose_key = f"loanpurpose_{loan_purpose.lower()}"
    if purpose_key in input_data:
        input_data[purpose_key] = 1

    # Build dataframe in the exact column order the model was trained on
    input_df = pd.DataFrame([input_data])[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### 📊 Result")
    if prediction == 1:
        st.error(f"⚠️ **High Risk — Likely to DEFAULT**")
    else:
        st.success(f"✅ **Low Risk — Likely to REPAY**")

    st.metric("Predicted Probability of Default", f"{probability * 100:.2f}%")
    st.progress(min(int(probability * 100), 100))

    with st.expander("See scaled input values sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))
