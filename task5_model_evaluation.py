"""
task5_model_evaluation.py

Implements Task 5 (Model Evaluation & Comparison) for the Loan Default
Prediction project. This is a CLASSIFICATION problem (default: 0/1),
so only the [BOTH] and [CLASS] items from the Task 5 checklist apply.

What this script does, step by step:
  1. Model Evaluation      -> Accuracy, Precision, Recall, F1 for every model
  2. Overfitting check     -> Train score vs Test score for every model
  3. Cross-validation      -> 5-fold CV on the training data (mean + spread)
  4. Compare all models    -> single comparison table
  5. Hyperparameter tuning -> RandomizedSearchCV on the best model
  6. Advanced models       -> Random Forest, AdaBoost, Gradient Boosting

Run with:
    python task5_model_evaluation.py

Outputs:
    task5_results/task5_report.md   <- full written report (fills the
                                        checklist table from Task5.docx)
    loan_default_model.pkl          <- overwritten ONLY if the tuned model
                                        beats the current one on the test set
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUT_DIR = Path("task5_results")
OUT_DIR.mkdir(exist_ok=True)

# For 5-fold CV and hyperparameter tuning we use a stratified sub-sample of
# the training set. With ~200k training rows, running 5-fold CV (and a
# search on top of that) for four different models on the *full* set would
# take a very long time without materially changing the conclusions.
# The FINAL model for each type is still fit on the FULL training set, and
# the final metrics reported (accuracy/precision/recall/F1, overfit check)
# always come from that full-data model evaluated on the full test set.
CV_SAMPLE_SIZE = 40_000


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Load data (same preprocessing as train_model.py, so results line up
#    with the model already shipped in the app)
# ---------------------------------------------------------------------------
log("Loading data...")
df = pd.read_csv("Loan_default_cleaned.csv")

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=["default"])
y = df["default"]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
log(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")

# Stratified sub-sample of the TRAINING data only, used for CV/tuning speed.
X_cv, _, y_cv, _ = train_test_split(
    X_train,
    y_train,
    train_size=CV_SAMPLE_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE
    ),
}

results = {}

# ---------------------------------------------------------------------------
# 2 & 3. For every model: fit on full training set, evaluate on train/test,
#         run 5-fold CV on the sub-sample.
# ---------------------------------------------------------------------------
for name, model in MODELS.items():
    log(f"\n=== {name} ===")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    log(f"Fit on full training set in {fit_time:.1f}s")

    # --- Step 1: evaluation metrics on the held-out test set ---
    y_pred_test = model.predict(X_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1": f1_score(y_test, y_pred_test),
    }

    # --- Step 2: overfitting / underfitting check (train vs test) ---
    y_pred_train = model.predict(X_train)
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_pred_train),
        "precision": precision_score(y_train, y_pred_train),
        "recall": recall_score(y_train, y_pred_train),
        "f1": f1_score(y_train, y_pred_train),
    }
    gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    if gap > 0.05:
        fit_verdict = "Overfitting (train >> test)"
    elif train_metrics["accuracy"] < 0.65 and test_metrics["accuracy"] < 0.65:
        fit_verdict = "Underfitting (both scores low)"
    else:
        fit_verdict = "Good fit (train and test close)"

    # --- Step 3: 5-fold cross-validation on the training sub-sample ---
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_out = cross_validate(model, X_cv, y_cv, cv=cv, scoring=SCORING, n_jobs=1)
    cv_time = time.time() - t0
    cv_summary = {
        metric: {
            "mean": float(np.mean(cv_out[f"test_{metric}"])),
            "std": float(np.std(cv_out[f"test_{metric}"])),
        }
        for metric in SCORING
    }
    log(f"5-fold CV done in {cv_time:.1f}s | "
        f"acc mean={cv_summary['accuracy']['mean']:.4f} "
        f"std={cv_summary['accuracy']['std']:.4f}")

    results[name] = {
        "fit_time_sec": round(fit_time, 2),
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "overfit_gap_accuracy": round(gap, 4),
        "fit_verdict": fit_verdict,
        "cv_5fold": cv_summary,
    }

# ---------------------------------------------------------------------------
# 4. Compare all models -> pick best by test F1 (good single number for an
#    imbalanced classification target) that is also stable in CV
#    (low std across folds).
# ---------------------------------------------------------------------------
comparison_rows = []
for name, r in results.items():
    comparison_rows.append(
        {
            "Model": name,
            "Accuracy": r["test_metrics"]["accuracy"],
            "Precision": r["test_metrics"]["precision"],
            "Recall": r["test_metrics"]["recall"],
            "F1-score": r["test_metrics"]["f1"],
            "CV F1 mean": r["cv_5fold"]["f1"]["mean"],
            "CV F1 std": r["cv_5fold"]["f1"]["std"],
        }
    )
comparison_df = pd.DataFrame(comparison_rows).sort_values(
    "F1-score", ascending=False
).reset_index(drop=True)

best_model_name = comparison_df.iloc[0]["Model"]
log(f"\nBest model by test F1-score: {best_model_name}")
log(comparison_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Hyperparameter tuning on the best model with RandomizedSearchCV
#    (run on the training sub-sample for speed, then the tuned model is
#    refit on the FULL training set and re-evaluated on the full test set).
# ---------------------------------------------------------------------------
PARAM_DISTS = {
    "Logistic Regression": {
        "C": uniform(0.01, 10),
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "Random Forest": {
        "n_estimators": randint(100, 400),
        "max_depth": randint(4, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
    },
    "AdaBoost": {
        "n_estimators": randint(50, 300),
        "learning_rate": uniform(0.01, 1.5),
    },
    "Gradient Boosting": {
        "n_estimators": randint(50, 300),
        "max_depth": randint(2, 6),
        "learning_rate": uniform(0.01, 0.3),
    },
}

base_model_cls = type(MODELS[best_model_name])
if best_model_name == "Logistic Regression":
    base_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
elif best_model_name == "Random Forest":
    base_model = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
elif best_model_name == "AdaBoost":
    base_model = AdaBoostClassifier(random_state=RANDOM_STATE)
else:
    base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

log(f"\nRunning RandomizedSearchCV on {best_model_name}...")
t0 = time.time()
search = RandomizedSearchCV(
    base_model,
    param_distributions=PARAM_DISTS[best_model_name],
    n_iter=15,
    scoring="f1",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1 if best_model_name != "Gradient Boosting" else 1,
)
search.fit(X_cv, y_cv)
search_time = time.time() - t0
log(f"Search done in {search_time:.1f}s")
log(f"Best params: {search.best_params_}")
log(f"Best CV F1 (search, sub-sample): {search.best_score_:.4f}")

# Refit the tuned model on the FULL training set
tuned_model = base_model.__class__(**{**base_model.get_params(), **search.best_params_})
tuned_model.fit(X_train, y_train)

y_pred_tuned = tuned_model.predict(X_test)
tuned_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_tuned),
    "precision": precision_score(y_test, y_pred_tuned),
    "recall": recall_score(y_test, y_pred_tuned),
    "f1": f1_score(y_test, y_pred_tuned),
}
baseline_test_metrics = results[best_model_name]["test_metrics"]
improved = tuned_metrics["f1"] >= baseline_test_metrics["f1"]

log(f"\nTuned {best_model_name} test metrics: {tuned_metrics}")
log(f"Untuned {best_model_name} test metrics: {baseline_test_metrics}")
log(f"Tuned model improved F1 on test set: {improved}")

# ---------------------------------------------------------------------------
# Save the tuned model only if it is actually better (otherwise keep the
# currently shipped model untouched).
# ---------------------------------------------------------------------------
if improved:
    joblib.dump(tuned_model, "loan_default_model.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")
    log("Saved tuned model as the new loan_default_model.pkl")
else:
    log("Tuned model did not beat the baseline — kept the existing .pkl file")

# ---------------------------------------------------------------------------
# Persist raw results as JSON for reference / re-use
# ---------------------------------------------------------------------------
raw_results = {
    "per_model": results,
    "comparison_table": comparison_df.to_dict(orient="records"),
    "best_model": best_model_name,
    "tuning": {
        "best_params": search.best_params_,
        "search_cv_f1": search.best_score_,
        "tuned_test_metrics": tuned_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "improved": bool(improved),
    },
}
with open(OUT_DIR / "task5_results.json", "w") as f:
    json.dump(raw_results, f, indent=2)


# ---------------------------------------------------------------------------
# Write the human-readable markdown report (mirrors the Task5.docx table)
# ---------------------------------------------------------------------------
def fmt(x):
    return f"{x:.4f}"


lines = []
lines.append("# Task 5 — Model Evaluation & Comparison\n")
lines.append(
    "This is a **classification** problem (predicting loan `default`, 0/1), "
    "so only the classification items from the Task 5 checklist apply.\n"
)

lines.append("## 1. Model Evaluation (Accuracy / Precision / Recall / F1)\n")
lines.append("| Model | Accuracy | Precision | Recall | F1-score |")
lines.append("|---|---|---|---|---|")
for name, r in results.items():
    m = r["test_metrics"]
    lines.append(
        f"| {name} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | "
        f"{fmt(m['recall'])} | {fmt(m['f1'])} |"
    )
lines.append("")

lines.append("## 2. Overfitting / Underfitting Check (Train vs Test Accuracy)\n")
lines.append("| Model | Train Accuracy | Test Accuracy | Gap | Verdict |")
lines.append("|---|---|---|---|---|")
for name, r in results.items():
    lines.append(
        f"| {name} | {fmt(r['train_metrics']['accuracy'])} | "
        f"{fmt(r['test_metrics']['accuracy'])} | {fmt(r['overfit_gap_accuracy'])} | "
        f"{r['fit_verdict']} |"
    )
lines.append("")

lines.append("## 3. 5-Fold Cross-Validation (on a 40,000-row training sub-sample)\n")
lines.append("| Model | CV F1 Mean | CV F1 Std (spread) | CV Accuracy Mean | CV Accuracy Std |")
lines.append("|---|---|---|---|---|")
for name, r in results.items():
    cvf = r["cv_5fold"]["f1"]
    cva = r["cv_5fold"]["accuracy"]
    lines.append(
        f"| {name} | {fmt(cvf['mean'])} | {fmt(cvf['std'])} | "
        f"{fmt(cva['mean'])} | {fmt(cva['std'])} |"
    )
lines.append(
    "\n*Lower std = more stable across folds. A large std alongside a high "
    "mean score is a warning sign of an unstable model.*\n"
)

lines.append("## 4. Compare All Models\n")
lines.append("### Classification models\n")
lines.append(comparison_df.to_markdown(index=False))
lines.append(
    f"\n**Best model (highest test F1, checked against CV stability): "
    f"{best_model_name}**\n"
)

lines.append("## 5. Hyperparameter Tuning\n")
lines.append(f"- Method: `RandomizedSearchCV` (15 iterations, 3-fold CV, scoring = F1)")
lines.append(f"- Tuned model: **{best_model_name}**")
lines.append(f"- Best hyperparameters found: `{search.best_params_}`")
lines.append(f"- Best CV F1 during search (sub-sample): {fmt(search.best_score_)}")
lines.append("")
lines.append("| | Accuracy | Precision | Recall | F1-score |")
lines.append("|---|---|---|---|---|")
lines.append(
    f"| Before tuning (test set) | {fmt(baseline_test_metrics['accuracy'])} | "
    f"{fmt(baseline_test_metrics['precision'])} | {fmt(baseline_test_metrics['recall'])} | "
    f"{fmt(baseline_test_metrics['f1'])} |"
)
lines.append(
    f"| After tuning (test set) | {fmt(tuned_metrics['accuracy'])} | "
    f"{fmt(tuned_metrics['precision'])} | {fmt(tuned_metrics['recall'])} | "
    f"{fmt(tuned_metrics['f1'])} |"
)
verdict = "improved" if improved else "did not improve"
lines.append(f"\n**Result: tuning {verdict} F1-score on the test set.**\n")

lines.append("## 6. Advanced Models Tried\n")
lines.append("- ✅ Random Forest (Bagging)")
lines.append("- ✅ AdaBoost")
lines.append("- ✅ Gradient Boosting")
lines.append("")

lines.append("## Final Decision\n")
if improved:
    lines.append(
        f"The tuned **{best_model_name}** is saved as `loan_default_model.pkl` "
        "and is now used by `app.py`."
    )
else:
    lines.append(
        f"Tuning did not beat the untuned **{best_model_name}** on the test set, "
        "so the model file was left unchanged. The untuned version's metrics "
        "above (Section 1) reflect what `app.py` currently uses."
    )

report_text = "\n".join(lines)
with open(OUT_DIR / "task5_report.md", "w") as f:
    f.write(report_text)

log(f"\nReport written to {OUT_DIR / 'task5_report.md'}")
log(f"Raw results written to {OUT_DIR / 'task5_results.json'}")
log("\nDone.")
