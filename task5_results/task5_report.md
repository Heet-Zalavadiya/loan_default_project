# Task 5 — Model Evaluation & Comparison

This is a **classification** problem (predicting loan `default`, 0/1), so only the classification items from the Task 5 checklist apply.

## 1. Model Evaluation (Accuracy / Precision / Recall / F1)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.6764 | 0.2196 | 0.6997 | 0.3343 |
| Random Forest | 0.7765 | 0.2705 | 0.5446 | 0.3615 |
| AdaBoost | 0.8858 | 0.6235 | 0.0430 | 0.0804 |
| Gradient Boosting | 0.8862 | 0.6261 | 0.0494 | 0.0916 |

## 2. Overfitting / Underfitting Check (Train vs Test Accuracy)

| Model | Train Accuracy | Test Accuracy | Gap | Verdict |
|---|---|---|---|---|
| Logistic Regression | 0.6752 | 0.6764 | -0.0012 | Good fit (train and test close) |
| Random Forest | 0.8183 | 0.7765 | 0.0418 | Good fit (train and test close) |
| AdaBoost | 0.8858 | 0.8858 | -0.0001 | Good fit (train and test close) |
| Gradient Boosting | 0.8867 | 0.8862 | 0.0006 | Good fit (train and test close) |

## 3. 5-Fold Cross-Validation (on a 40,000-row training sub-sample)

| Model | CV F1 Mean | CV F1 Std (spread) | CV Accuracy Mean | CV Accuracy Std |
|---|---|---|---|---|
| Logistic Regression | 0.3303 | 0.0049 | 0.6762 | 0.0031 |
| Random Forest | 0.2951 | 0.0183 | 0.8566 | 0.0048 |
| AdaBoost | 0.0861 | 0.0139 | 0.8856 | 0.0008 |
| Gradient Boosting | 0.1018 | 0.0060 | 0.8863 | 0.0007 |

*Lower std = more stable across folds. A large std alongside a high mean score is a warning sign of an unstable model.*

## 4. Compare All Models

### Classification models

| Model               |   Accuracy |   Precision |    Recall |   F1-score |   CV F1 mean |   CV F1 std |
|:--------------------|-----------:|------------:|----------:|-----------:|-------------:|------------:|
| Random Forest       |   0.776542 |    0.270497 | 0.544596  |  0.361459  |    0.295122  |  0.0183473  |
| Logistic Regression |   0.676366 |    0.219612 | 0.699713  |  0.3343    |    0.330284  |  0.00490977 |
| Gradient Boosting   |   0.886176 |    0.626068 | 0.0494015 |  0.0915768 |    0.101814  |  0.00604114 |
| AdaBoost            |   0.885843 |    0.623472 | 0.0429944 |  0.0804416 |    0.0861422 |  0.0138874  |

**Best model (highest test F1, checked against CV stability): Random Forest**

## 5. Hyperparameter Tuning

- Method: `RandomizedSearchCV` (15 iterations, 3-fold CV, scoring = F1)
- Tuned model: **Random Forest**
- Best hyperparameters found: `{'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 16, 'n_estimators': 206}`
- Best CV F1 during search (sub-sample): 0.3490

| | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Before tuning (test set) | 0.7765 | 0.2705 | 0.5446 | 0.3615 |
| After tuning (test set) | 0.7298 | 0.2451 | 0.6378 | 0.3541 |

**Result: tuning did not improve F1-score on the test set.**

## 6. Advanced Models Tried

- ✅ Random Forest (Bagging)
- ✅ AdaBoost
- ✅ Gradient Boosting

## Final Decision

Tuning did not beat the untuned **Random Forest** on the test set, so the model file was left unchanged. The untuned version's metrics above (Section 1) reflect what `app.py` currently uses.