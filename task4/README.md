# Task 4 - Evaluation and Tuning

## Project Overview

This project evaluates machine learning classification and regression models for a banking loan prediction scenario.

The objective is to evaluate model performance, tune hyperparameters, select the best model and perform final testing.

## Dataset

A synthetic loan dataset containing 1000 records was created.

Features:
- Age
- Income
- Loan Amount
- Credit Score
- Employment Years
- Gender
- Property Area

Target:
- Loan Default (0 = No Default, 1 = Default)

## 1. Classification Evaluation

Three classification models were trained:

- Logistic Regression
- Random Forest
- XGBoost

The following metrics were calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Validation Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.665 | 0.5000 | 0.2090 | 0.2947 |
| Random Forest | 0.740 | 0.7273 | 0.3582 | 0.4800 |
| XGBoost | 0.740 | 0.7027 | 0.3881 | 0.5000 |
| Tuned Random Forest | 0.760 | 0.7879 | 0.3881 | 0.5200 |
| Tuned XGBoost | 0.705 | 0.5833 | 0.4179 | 0.4870 |

### Why Accuracy Can Be Misleading

Accuracy can be misleading when the target classes are imbalanced.

A model may predict the majority class most of the time and still achieve high accuracy while failing to identify many minority-class examples.

Precision, recall, F1 score and confusion matrices provide a more complete understanding of model performance.

## 2. ROC Curves and AUC

ROC curves were plotted for three classification models.

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.7300 |
| Random Forest | 0.7931 |
| XGBoost | 0.7844 |

The ROC curve compares the true positive rate with the false positive rate at different classification thresholds.

The generated plot is saved as:

`roc_curves.png`

## 3. Decision Threshold Tuning

The classification threshold was adjusted to achieve at least 75% recall.

Results:

- Selected Model: Random Forest
- Threshold: 0.3333
- Precision: 0.5778
- Recall: 0.7761

The selected threshold satisfies the required recall of at least 75%.

## 4. Hyperparameter Tuning

### Random Forest - GridSearchCV

Three parameters were tuned:

- n_estimators
- max_depth
- min_samples_split

Best Parameters:

- max_depth: None
- min_samples_split: 2
- n_estimators: 100

Best Cross-Validation F1 Score: 0.4645

### XGBoost - RandomizedSearchCV

RandomizedSearchCV was performed with 15 iterations.

Parameters tuned:

- n_estimators
- max_depth
- learning_rate

Best Parameters:

- n_estimators: 150
- max_depth: 6
- learning_rate: 0.2

Best Cross-Validation F1 Score: 0.4448

## 5. Best Model Selection

The best model was selected using validation ROC-AUC.

Selected Model: Tuned Random Forest

- Validation ROC-AUC: 0.8009
- Validation F1 Score: 0.5200

## 6. Regression Evaluation

Linear Regression was used to predict loan amount.

The model was compared against a mean prediction baseline.

| Metric | Linear Regression | Mean Baseline |
|---|---:|---:|
| MAE | 10878.46 | 17242.99 |
| RMSE | 13701.62 | 21761.66 |
| R² Score | 0.6021 | -0.0038 |

Linear Regression performed better than the mean baseline.

Lower MAE and RMSE indicate better prediction accuracy, while a higher R² indicates that the model explains more variation in the target.

## 7. Final Test Set Evaluation

The selected Tuned Random Forest model was evaluated on the held-out test set.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.6450 |
| Precision | 0.4545 |
| Recall | 0.2985 |
| F1 Score | 0.3604 |
| ROC-AUC | 0.6636 |

Confusion Matrix:

```text
[[109  24]
 [ 47  20]]
```

The test set was kept separate from training, model tuning and validation-based model selection.

The final test metrics provide an independent estimate of model performance.

## 8. Output Files

- `main.py` - Complete Python implementation
- `README.md` - Project documentation
- `roc_curves.png` - ROC curve visualization
- `classification_results.csv` - Classification metrics for all evaluated models

## 9. Concepts Covered

- Classification Metrics
- Confusion Matrix
- Accuracy, Precision, Recall and F1 Score
- ROC Curve and ROC-AUC
- Decision Threshold Tuning
- GridSearchCV
- RandomizedSearchCV
- Regression Metrics
- Mean Baseline Comparison
- Final Test Evaluation

## Conclusion

Task 4 successfully demonstrates model evaluation, threshold selection, hyperparameter tuning, regression comparison and final test evaluation.

The tuned Random Forest achieved the highest validation ROC-AUC among the evaluated candidates.

TASK 4 COMPLETED SUCCESSFULLY.