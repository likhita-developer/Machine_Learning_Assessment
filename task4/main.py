import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. CREATE LOAN DATASET
# ==========================================

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 150000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 30, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "property_area": np.random.choice(
        ["Urban", "Rural", "Semiurban"], n
    )
})

# Create loan amount using other features
noise = np.random.normal(0, 12000, n)

df["loan_amount"] = (
    0.45 * df["income"]
    + 350 * df["age"]
    + 35 * df["credit_score"]
    + 900 * df["employment_years"]
    + noise
)

df["loan_amount"] = df["loan_amount"].clip(10000, 200000)

# Create classification target
risk = (
    0.15
    + 0.35 * (df["credit_score"] < 550)
    + 0.25 * (df["income"] < 40000)
    + 0.15 * (df["loan_amount"] > 150000)
)

risk = np.clip(risk, 0.05, 0.95)

df["loan_default"] = np.random.binomial(1, risk)

# Add missing values
for column in ["income", "credit_score", "gender"]:
    missing_indices = df.sample(
        frac=0.03,
        random_state=42
    ).index

    df.loc[missing_indices, column] = np.nan

print("DATASET SHAPE:", df.shape)

print("\nCLASS DISTRIBUTION:")
print(df["loan_default"].value_counts())


# ==========================================
# 2. TRAIN, VALIDATION AND TEST SPLIT
# ==========================================

X = df.drop(columns=["loan_default"])
y = df["loan_default"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nTRAIN:", X_train.shape)
print("VALIDATION:", X_val.shape)
print("TEST:", X_test.shape)


# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================

numeric_features = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years"
]

categorical_features = [
    "gender",
    "property_area"
]


def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


# ==========================================
# 4. TRAIN THREE CLASSIFIERS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist"
    )
}

pipelines = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    pipelines[name] = pipeline


# ==========================================
# 5. CLASSIFICATION EVALUATION
# ==========================================

print("\nCLASSIFICATION METRICS")
print("=" * 60)

for name, pipeline in pipelines.items():

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print("\nMODEL:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))


# Explain accuracy and class imbalance
print("\nWHY ACCURACY CAN BE MISLEADING")
print(
    "When classes are imbalanced, a model can predict "
    "the majority class most of the time and still show "
    "high accuracy while missing many minority-class cases."
)

print(
    "Precision, recall, F1 score and the confusion matrix "
    "provide a more complete evaluation."
)


# ==========================================
# 6. ROC CURVES AND AUC
# ==========================================

plt.figure(figsize=(9, 6))

auc_scores = {}

for name, pipeline in pipelines.items():

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_val, probabilities
    )

    auc = roc_auc_score(y_val, probabilities)
    auc_scores[name] = auc

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Model"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Loan Default Prediction")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.close()

print("\nROC CURVES SAVED: roc_curves.png")
print("AUC SCORES:", auc_scores)


# ==========================================
# 7. DECISION THRESHOLD TUNING
# ==========================================

print("\nDECISION THRESHOLD TUNING")

best_threshold = None
best_precision = -1
best_recall = 0
threshold_model = None

for name, pipeline in pipelines.items():

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(
        y_val, probabilities
    )

    valid_indices = np.where(recalls[:-1] >= 0.75)[0]

    if len(valid_indices) > 0:

        index = valid_indices[
            np.argmax(precisions[valid_indices])
        ]

        if precisions[index] > best_precision:

            best_precision = precisions[index]
            best_recall = recalls[index]
            best_threshold = thresholds[index]
            threshold_model = name

if best_threshold is not None:

    print("Selected Model:", threshold_model)
    print("Selected Threshold:", round(best_threshold, 4))
    print("Precision:", round(best_precision, 4))
    print("Recall:", round(best_recall, 4))

else:
    print("No model achieved 75% recall on validation data.")


# ==========================================
# 8. GRID SEARCH - RANDOM FOREST
# ==========================================

print("\nRANDOM FOREST GRID SEARCH")

rf_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            n_jobs=1
        )
    )
])

rf_parameters = {
    "classifier__n_estimators": [100, 150],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    cv=3,
    scoring="f1",
    n_jobs=1
)

rf_grid.fit(X_train, y_train)

print("Best Parameters:", rf_grid.best_params_)
print("Best F1 Score:", round(rf_grid.best_score_, 4))


# ==========================================
# 9. RANDOMIZED SEARCH - XGBOOST
# ==========================================

print("\nXGBOOST RANDOMIZED SEARCH")

xgb_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),

    (
        "classifier",
        XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            tree_method="hist"
        )
    )
])

xgb_parameters = {
    "classifier__n_estimators": [100, 150, 200, 250],
    "classifier__max_depth": [3, 4, 5, 6],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_parameters,
    n_iter=15,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=1
)

xgb_random.fit(X_train, y_train)

print("Best Parameters:", xgb_random.best_params_)
print("Best F1 Score:", round(xgb_random.best_score_, 4))


# ==========================================
# 10. SELECT BEST MODEL USING VALIDATION AUC
# ==========================================

candidate_models = {
    **pipelines,
    "Tuned Random Forest": rf_grid.best_estimator_,
    "Tuned XGBoost": xgb_random.best_estimator_
}

best_name = None
best_model = None
best_auc = -1
best_f1 = 0

for name, model in candidate_models.items():

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = model.predict(X_val)

    auc = roc_auc_score(y_val, probabilities)

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    if auc > best_auc:

        best_auc = auc
        best_f1 = f1
        best_name = name
        best_model = model

print("\nBEST MODEL SELECTED")
print("Best Model:", best_name)
print("Validation ROC-AUC:", round(best_auc, 4))
print("Validation F1:", round(best_f1, 4))


# ==========================================
# 11. REGRESSION MODEL VS MEAN BASELINE
# ==========================================

print("\nREGRESSION MODEL EVALUATION")

regression_features = [
    "age",
    "income",
    "credit_score",
    "employment_years"
]

regression_X = df[regression_features]
regression_y = df["loan_amount"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    regression_X,
    regression_y,
    test_size=0.20,
    random_state=42
)

regression_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", __import__(
        "sklearn.linear_model",
        fromlist=["LinearRegression"]
    ).LinearRegression())
])

regression_model.fit(Xr_train, yr_train)

regression_predictions = regression_model.predict(Xr_test)

mean_predictions = np.full(
    len(yr_test),
    yr_train.mean()
)

for name, predictions in [
    ("Linear Regression", regression_predictions),
    ("Mean Baseline", mean_predictions)
]:

    mae = mean_absolute_error(yr_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(yr_test, predictions)
    )

    r2 = r2_score(yr_test, predictions)

    print("\n", name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))


# ==========================================
# 12. FINAL TEST SET EVALUATION
# ==========================================

print("\nFINAL TEST SET EVALUATION")
print("Selected Model:", best_name)

# Evaluate the selected classifier on the test set once
test_predictions = best_model.predict(X_test)

test_probabilities = best_model.predict_proba(X_test)[:, 1]

print("Test Accuracy:", round(
    accuracy_score(y_test, test_predictions), 4
))

print("Test Precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 4
))

print("Test Recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 4
))

print("Test F1 Score:", round(
    f1_score(y_test, test_predictions, zero_division=0), 4
))

print("Test ROC-AUC:", round(
    roc_auc_score(y_test, test_probabilities), 4
))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# ==========================================
# 13. SAVE RESULTS
# ==========================================

results = []

for name, pipeline in candidate_models.items():

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_val, probabilities)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "classification_results.csv",
    index=False
)

print("\nCLASSIFICATION RESULTS SAVED")
print(results_df)

print("\nTASK 4 EXECUTION COMPLETED SUCCESSFULLY!")