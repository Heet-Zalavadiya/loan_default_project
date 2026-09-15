# Loan Default Prediction — Streamlit App

A web app that predicts whether a loan applicant is likely to **default**,
powered by a Logistic Regression model.

## 📁 Files
| File | Purpose |
|---|---|
| `Loan_default_cleaned.csv` | Your feature-scaled training dataset |
| `train_model.py` | Trains the Logistic Regression model and saves it |
| `loan_default_model.pkl` | Saved trained model (created by `train_model.py`) |
| `feature_columns.pkl` | Saved feature column order (created by `train_model.py`) |
| `app.py` | The Streamlit web app |
| `requirements.txt` | Python dependencies |

## 🚀 Step-by-Step Setup

### 1. Create a project folder and place these files inside it
Make sure `Loan_default_cleaned.csv` is in the same folder.

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (run once)
This reads `Loan_default_cleaned.csv`, trains the Logistic Regression model,
and saves `loan_default_model.pkl` + `feature_columns.pkl`.
```bash
python train_model.py
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

## 🖱️ How to Use the App
1. Fill in the applicant's **Personal Information** (age, income, education, etc.)
2. Fill in the **Loan Details** (loan amount, interest rate, term, DTI ratio, etc.)
3. Select **Other Factors** (mortgage, dependents, co-signer, loan purpose)
4. Click **🔍 Predict Default Risk**
5. View:
   - Prediction (Likely to Default / Likely to Repay)
   - Predicted probability of default (%)
   - (Optional) The scaled values sent to the model, in an expandable section

## ⚙️ How Scaling Works
Your CSV was already Min-Max scaled to [0, 1] before training. Since real users
enter raw values (e.g., age = 35, income = $50,000), `app.py` re-applies the
same Min-Max scaling formula using the known original ranges of this dataset
(age 18–69, income 15,000–149,999, loan amount 5,000–249,999, credit score
300–849, months employed 0–119, credit lines 1–4, interest rate 2–25%,
loan term ∈ {12, 24, 36, 48, 60}, DTI ratio 0.1–0.9) before passing the input to
the model — so predictions stay consistent with how the model was trained.

## 🔁 Retraining
If you get a new version of the dataset, just rerun `python train_model.py`
to regenerate the `.pkl` files — the app will automatically pick up the new model.

## ✅ Task 5 — Model Evaluation & Comparison
`task5_model_evaluation.py` implements the full Task 5 checklist for this
(classification) problem:

1. **Model Evaluation** — Accuracy, Precision, Recall, F1-score for every model
2. **Overfitting / Underfitting check** — train vs test score for every model
3. **5-Fold Cross-Validation** — mean score and spread (stability) per model
4. **Compare All Models** — Logistic Regression, Random Forest, AdaBoost,
   Gradient Boosting, in one table
5. **Hyperparameter Tuning** — `RandomizedSearchCV` on the best model
   (Random Forest), then re-tested on the test set
6. **Advanced Models** — Random Forest (Bagging), AdaBoost, Gradient Boosting

Run it with:
```bash
python task5_model_evaluation.py
```
It writes a full report to `task5_results/task5_report.md` (and raw numbers
to `task5_results/task5_results.json`). It only overwrites
`loan_default_model.pkl` if the tuned model actually beats the current one
on the test set — in the run included with this project, tuning did **not**
beat the baseline Random Forest, so `app.py` still uses the original
untuned model. See `task5_report.md` and the completed `Task5.docx` for full
results and reasoning.
