"""
app.py
Streamlit web app: Loan Default Prediction & ML Analytics Dashboard.
Run with:  streamlit run app.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page config & Custom CSS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Default Risk & ML Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ------------------------------------------------------------------
# Load artifacts & cached data
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_default_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

@st.cache_data
def load_task5_results():
    json_path = os.path.join("task5_results", "task5_results.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_dataset_sample():
    csv_path = "Loan_default_cleaned.csv"
    if os.path.exists(csv_path):
        # Load sample to keep app fast
        df = pd.read_csv(csv_path)
        return df
    return None

model, feature_columns = load_artifacts()
task5_results = load_task5_results()
df_dataset = load_dataset_sample()

# Min-max ranges for numeric scaling
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
LOAN_TERMS = [12, 24, 36, 48, 60]

def scale(value, col):
    lo, hi = RANGES[col]
    return (value - lo) / (hi - lo)

def scale_loan_term(term):
    return (term - min(LOAN_TERMS)) / (max(LOAN_TERMS) - min(LOAN_TERMS))


# ------------------------------------------------------------------
# Sidebar & Navigation Header
# ------------------------------------------------------------------
st.sidebar.title("🏦 Loan Risk AI")
st.sidebar.caption("Machine Learning Credit Risk Assessment")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Overview")
st.sidebar.write(
    "This web application predicts loan default risk using trained classification models "
    "and presents complete model evaluation metrics"
)

if task5_results and "per_model" in task5_results:
    lr_acc = task5_results["per_model"]["Logistic Regression"]["test_metrics"]["accuracy"]
    gb_acc = task5_results["per_model"]["Gradient Boosting"]["test_metrics"]["accuracy"]
    rf_acc = task5_results["per_model"]["Random Forest"]["test_metrics"]["accuracy"]
    st.sidebar.metric("Logistic Reg Accuracy", f"{lr_acc*100:.2f}%")
    st.sidebar.metric("Top Model Accuracy (GB)", f"{gb_acc*100:.2f}%")
    st.sidebar.metric("Best F1 Model (RF)", f"{rf_acc*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use tabs at the top to navigate between Live Risk Prediction, Model Accuracy, Feature Importance, and Dataset Insights.")


# Header
st.title("💰 Loan Default Risk Predictor")
st.markdown("Assess loan applicant risk in real-time and explore model performance, accuracy metrics, and key risk drivers.")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Risk Predictor", 
    "📊 Model Accuracy & Benchmarks", 
    "📈 Risk Drivers & Coefficients", 
    "📁 Dataset Insights"
])


# ==================================================================
# TAB 1: Live Risk Predictor
# ==================================================================
with tab1:
    st.subheader("👤 Enter Applicant Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Personal Profile")
        age = st.number_input("Age", min_value=18, max_value=69, value=35)
        income = st.number_input("Annual Income ($)", min_value=15000, max_value=149999, value=55000, step=1000)
        education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        months_employed = st.number_input("Months Employed", min_value=0, max_value=119, value=36)

    with col2:
        st.markdown("### Loan Profile")
        loan_amount = st.number_input("Loan Amount ($)", min_value=5000, max_value=249999, value=25000, step=500)
        interest_rate = st.number_input("Interest Rate (%)", min_value=2.0, max_value=25.0, value=10.5, step=0.1)
        loan_term = st.selectbox("Loan Term (Months)", LOAN_TERMS, index=2)
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.1, max_value=0.9, value=0.35, step=0.01)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=1, max_value=4, value=2)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=849, value=670)

    st.markdown("### 📌 Additional Risk Factors")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        has_mortgage = st.radio("Has Mortgage?", ["Yes", "No"], horizontal=True)
    with col4:
        has_dependents = st.radio("Has Dependents?", ["Yes", "No"], horizontal=True)
    with col5:
        has_cosigner = st.radio("Has Co-signer?", ["Yes", "No"], horizontal=True)
    with col6:
        loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

    st.markdown("---")

    if st.button("🔍 Evaluate Loan Risk", use_container_width=True):
        # Prepare input dict
        input_data = {col: 0 for col in feature_columns}

        input_data["age"] = scale(age, "age")
        input_data["income"] = scale(income, "income")
        input_data["loanamount"] = scale(loan_amount, "loanamount")
        input_data["creditscore"] = scale(credit_score, "creditscore")
        input_data["monthsemployed"] = scale(months_employed, "monthsemployed")
        input_data["numcreditlines"] = scale(num_credit_lines, "numcreditlines")
        input_data["interestrate"] = scale(interest_rate, "interestrate")
        input_data["loanterm"] = scale_loan_term(loan_term)
        input_data["dtiratio"] = scale(dti_ratio, "dtiratio")

        input_data["hasmortgage"] = 1 if has_mortgage == "Yes" else 0
        input_data["hasdependents"] = 1 if has_dependents == "Yes" else 0
        input_data["hascosigner"] = 1 if has_cosigner == "Yes" else 0

        edu_key = f"education_{education.lower().replace(' ', '_')}"
        if edu_key in input_data:
            input_data[edu_key] = 1

        emp_key = f"employmenttype_{employment_type.lower()}"
        if emp_key in input_data:
            input_data[emp_key] = 1

        mar_key = f"maritalstatus_{marital_status.lower()}"
        if mar_key in input_data:
            input_data[mar_key] = 1

        purpose_key = f"loanpurpose_{loan_purpose.lower()}"
        if purpose_key in input_data:
            input_data[purpose_key] = 1

        input_df = pd.DataFrame([input_data])[feature_columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("## 📊 Assessment Outcome")
        
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK — LIKELY TO DEFAULT**")
            else:
                st.success("✅ **LOW RISK — LIKELY TO REPAY**")

            st.metric(
                label="Predicted Default Probability", 
                value=f"{probability * 100:.2f}%", 
                delta=f"{'High Risk' if probability > 0.5 else 'Acceptable Risk'}", 
                delta_color="inverse"
            )
            st.progress(min(int(probability * 100), 100))

        with res_col2:
            # Interactive Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Probability Gauge (%)", 'font': {'size': 18, 'color': "#ffffff"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#ef4444" if probability > 0.5 else "#10b981"},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#ffffff"},
                height=220,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander("🔬 View Scaled Input Vector"):
            st.dataframe(input_df.T.rename(columns={0: "Scaled Value"}))


# ==================================================================
# TAB 2: Model Performance & Accuracy
# ==================================================================
with tab2:
    st.subheader("📊 Model Evaluation & Comparison Benchmarks")
    st.write("Results evaluated across 4 trained classification algorithms on the test dataset.")

    if task5_results and "comparison_table" in task5_results:
        comp_df = pd.DataFrame(task5_results["comparison_table"])
        
        # Display KPI metric cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        best_acc_model = comp_df.loc[comp_df["Accuracy"].idxmax()]
        best_f1_model = comp_df.loc[comp_df["F1-score"].idxmax()]
        best_rec_model = comp_df.loc[comp_df["Recall"].idxmax()]

        with kpi1:
            st.metric("Highest Test Accuracy", f"{best_acc_model['Accuracy']*100:.2f}%", f"{best_acc_model['Model']}")
        with kpi2:
            st.metric("Highest Precision", f"{comp_df['Precision'].max()*100:.2f}%", "Gradient Boosting")
        with kpi3:
            st.metric("Highest Recall", f"{best_rec_model['Recall']*100:.2f}%", f"{best_rec_model['Model']}")
        with kpi4:
            st.metric("Highest F1 Score", f"{best_f1_model['F1-score']*100:.2f}%", f"{best_f1_model['Model']}")

        st.markdown("---")

        # Interactive Model Comparison Charts
        st.markdown("### 📈 Model Evaluation Metrics Comparison")
        
        metrics_melted = comp_df.melt(
            id_vars=["Model"], 
            value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
            var_name="Metric", 
            value_name="Score"
        )
        metrics_melted["Score (%)"] = metrics_melted["Score"] * 100

        fig_comp = px.bar(
            metrics_melted,
            x="Model",
            y="Score (%)",
            color="Metric",
            barmode="group",
            title="Accuracy, Precision, Recall, & F1 Score Comparison",
            color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ec4899"],
            text_auto=".1f"
        )
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font={'color': '#f8fafc'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Classification Model",
            yaxis_title="Score (%)",
            height=420
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Detailed Comparison Table
        st.markdown("### 📋 Detailed Benchmark Summary Table")
        formatted_df = comp_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-score", "CV F1 mean", "CV F1 std"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        
        st.dataframe(formatted_df, use_container_width=True)

        # Cross Validation & Overfit Check
        st.markdown("---")
        st.markdown("### 🔄 5-Fold Cross-Validation & Overfitting Analysis")
        
        per_model = task5_results.get("per_model", {})
        cv_rows = []
        for name, mdata in per_model.items():
            cv_rows.append({
                "Model": name,
                "Train Acc": f"{mdata['train_metrics']['accuracy']*100:.2f}%",
                "Test Acc": f"{mdata['test_metrics']['accuracy']*100:.2f}%",
                "Overfit Gap": f"{mdata['overfit_gap_accuracy']*100:.2f}%",
                "Fit Verdict": mdata['fit_verdict'],
                "5-Fold CV Mean Acc": f"{mdata['cv_5fold']['accuracy']['mean']*100:.2f}%",
                "5-Fold CV Std": f"{mdata['cv_5fold']['accuracy']['std']*100:.2f}%"
            })
        st.table(pd.DataFrame(cv_rows))


# ==================================================================
# TAB 3: Feature Importance & Risk Factors
# ==================================================================
with tab3:
    st.subheader("📈 Feature Importance & Logistic Regression Coefficients")
    st.write("Understand which customer features have the strongest positive or negative effect on loan default prediction.")

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        feat_imp_df = pd.DataFrame({
            "Feature": feature_columns,
            "Coefficient": coefs,
            "AbsImpact": np.abs(coefs),
            "Impact Type": np.where(coefs > 0, "Increases Default Risk", "Reduces Default Risk")
        }).sort_values(by="Coefficient", ascending=True)

        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig_coef = px.bar(
                feat_imp_df,
                x="Coefficient",
                y="Feature",
                orientation="h",
                color="Impact Type",
                title="Model Coefficients (Feature Influence on Loan Default)",
                color_discrete_map={
                    "Increases Default Risk": "#ef4444",
                    "Reduces Default Risk": "#10b981"
                }
            )
            fig_coef.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                font={'color': '#f8fafc'},
                height=520,
                xaxis_title="Logistic Regression Coefficient Weight",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig_coef, use_container_width=True)

        with col_right:
            st.markdown("### 💡 Key Takeaways")
            st.markdown(
                "- **Positive Coefficients (Red):** Higher values increase default probability (e.g., Interest Rate, Debt-to-Income Ratio).\n"
                "- **Negative Coefficients (Green):** Higher values lower default risk (e.g., Credit Score, Income, Age).\n"
                "- **Categorical Factors:** One-hot encoded features reflect baseline shift relative to default class."
            )
            st.dataframe(
                feat_imp_df[["Feature", "Coefficient", "Impact Type"]].sort_values(by="Coefficient", ascending=False),
                height=360,
                use_container_width=True
            )


# ==================================================================
# TAB 4: Dataset Insights
# ==================================================================
with tab4:
    st.subheader("📁 Dataset Summary & Insights")
    
    if df_dataset is not None:
        st.write(f"Displaying statistics from `Loan_default_cleaned.csv` (**{len(df_dataset):,} Total Applicants**)")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total Records", f"{len(df_dataset):,}")
        with d2:
            default_cnt = df_dataset["default"].sum() if "default" in df_dataset.columns else 0
            st.metric("Defaulted Loans", f"{default_cnt:,}")
        with d3:
            default_rate = (default_cnt / len(df_dataset)) * 100 if len(df_dataset) > 0 else 0
            st.metric("Overall Default Rate", f"{default_rate:.2f}%")

        st.markdown("---")

        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### Target Class Distribution")
            if "default" in df_dataset.columns:
                class_counts = df_dataset["default"].value_counts().reset_index()
                class_counts.columns = ["Default Status", "Count"]
                class_counts["Label"] = class_counts["Default Status"].map({0: "Repaid (0)", 1: "Defaulted (1)"})

                fig_pie = px.pie(
                    class_counts,
                    names="Label",
                    values="Count",
                    hole=0.4,
                    color="Label",
                    color_discrete_map={"Repaid (0)": "#10b981", "Defaulted (1)": "#ef4444"},
                    title="Loan Default vs Repaid Ratio"
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#f8fafc'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.markdown("### Numeric Features Summary")
            num_cols = df_dataset.select_dtypes(include=[np.number]).columns
            st.dataframe(df_dataset[num_cols].describe().T[["mean", "std", "min", "50%", "max"]], height=320)

        st.markdown("### 🔍 Dataset Preview (First 50 Rows)")
        st.dataframe(df_dataset.head(50), use_container_width=True)
    else:
        st.warning("`Loan_default_cleaned.csv` was not found in the workspace.")
