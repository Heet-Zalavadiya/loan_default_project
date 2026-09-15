"""
train_model.py
Trains a Logistic Regression model on the pre-scaled loan default dataset
and saves the trained model + feature column order for use in the Streamlit app.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Load data
df = pd.read_csv("Loan_default_cleaned.csv")

# 2. Convert boolean one-hot columns to int
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# 3. Split features/target
X = df.drop(columns=["default"])
y = df["default"]

feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# 5. Save model + feature column order
joblib.dump(model, "loan_default_model.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("Model trained and saved successfully.")
print("Feature columns:", feature_columns)
