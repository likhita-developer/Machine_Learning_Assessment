# ==================================================
# TASK 3: SUPERVISED AND UNSUPERVISED LEARNING
# ==================================================

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, silhouette_score

from xgboost import XGBClassifier


# ==================================================
# 1. CREATE LOAN DATASET
# ==================================================

np.random.seed(42)

n = 500

df = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 150001, n),
    "loan_amount": np.random.randint(10000, 200001, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 30, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "property_area": np.random.choice(
        ["Urban", "Rural", "Semiurban"], n
    ),
    "education": np.random.choice(
        ["Undergraduate", "Graduate", "Postgraduate"], n
    )
})

default_probability = (
    0.15
    + 0.20 * (df["credit_score"] < 550)
    + 0.15 * (df["income"] < 40000)
    + 0.10 * (df["loan_amount"] > 150000)
)

default_probability = np.clip(default_probability, 0, 1)

df["loan_default"] = np.random.binomial(
    1, default_probability
)

print("\n========== LOAN DATASET ==========")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())


# ==================================================
# 2. TRAIN AND VALIDATION SPLIT
# ==================================================

X = df.drop("loan_default", axis=1)
y = df["loan_default"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years"
]

categorical_features = [
    "gender",
    "property_area",
    "education"
]

print("\nTRAINING DATA SHAPE:", X_train.shape)
print("VALIDATION DATA SHAPE:", X_val.shape)


# ==================================================
# 3. PREPROCESSING
# ==================================================

def create_preprocessor():

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


print("\nData preparation completed!")


# ==================================================
# 4. LOGISTIC REGRESSION
# ==================================================

print("\n========== LOGISTIC REGRESSION ==========")

logistic_model = Pipeline(steps=[
    ("preprocessor", create_preprocessor()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_accuracy = accuracy_score(
    y_val, logistic_predictions
)

print("Logistic Regression Validation Accuracy:",
      round(logistic_accuracy, 4))


# ==================================================
# 5. DECISION TREE
# ==================================================

print("\n========== DECISION TREE ==========")

decision_tree_model = Pipeline(steps=[
    ("preprocessor", create_preprocessor()),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

decision_tree_model.fit(X_train, y_train)

tree_predictions = decision_tree_model.predict(X_val)

tree_accuracy = accuracy_score(
    y_val, tree_predictions
)

print("Decision Tree Validation Accuracy:",
      round(tree_accuracy, 4))


# ==================================================
# 6. RANDOM FOREST
# ==================================================

print("\n========== RANDOM FOREST ==========")

random_forest_model = Pipeline(steps=[
    ("preprocessor", create_preprocessor()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])

random_forest_model.fit(X_train, y_train)

forest_predictions = random_forest_model.predict(X_val)

forest_accuracy = accuracy_score(
    y_val, forest_predictions
)

forest_classifier = random_forest_model.named_steps["classifier"]

print("Random Forest Validation Accuracy:",
      round(forest_accuracy, 4))

print("Random Forest OOB Score:",
      round(forest_classifier.oob_score_, 4))


# ==================================================
# 7. XGBOOST PIPELINE WITH EARLY STOPPING
# ==================================================

print("\n========== XGBOOST ==========")

# Fit a separate preprocessor on training data
xgb_preprocessor = create_preprocessor()

xgb_preprocessor.fit(X_train)

# Transform validation data using training-fitted preprocessing
X_val_processed = xgb_preprocessor.transform(X_val)

# Build XGBoost pipeline
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", xgb_preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42
    ))
])

# Train pipeline with validation data for early stopping
xgb_pipeline.fit(
    X_train,
    y_train,
    classifier__eval_set=[(X_val_processed, y_val)],
    classifier__verbose=False
)

xgb_model = xgb_pipeline.named_steps["classifier"]

xgb_predictions = xgb_pipeline.predict(X_val)

xgb_accuracy = accuracy_score(
    y_val, xgb_predictions
)

print("XGBoost Validation Accuracy:",
      round(xgb_accuracy, 4))

print("Best Iteration:", xgb_model.best_iteration)


# ==================================================
# 8. LOGISTIC REGRESSION PROBABILITIES
# ==================================================

print("\n========== PREDICT PROBABILITIES ==========")

sample_rows = X_val.iloc[:5]

probabilities = logistic_model.predict_proba(sample_rows)

predictions = logistic_model.predict(sample_rows)

probability_results = pd.DataFrame({
    "Actual": y_val.iloc[:5].values,
    "Predicted": predictions,
    "Probability_No_Default": probabilities[:, 0],
    "Probability_Default": probabilities[:, 1]
})

print(probability_results.round(4))


# ==================================================
# 9. REGRESSION: LINEAR, RIDGE AND LASSO
# ==================================================

print("\n========== REGRESSION MODELS ==========")

regression_features = [
    "age",
    "income",
    "credit_score",
    "employment_years"
]

X_reg = df[regression_features]
y_reg = df["loan_amount"]

X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

regression_results = []
coefficient_data = {}

for name, model in regression_models.items():

    regression_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", model)
    ])

    regression_pipeline.fit(
        X_reg_train,
        y_reg_train
    )

    predictions = regression_pipeline.predict(X_reg_val)

    score = regression_pipeline.score(
        X_reg_val,
        y_reg_val
    )

    coefficients = regression_pipeline.named_steps[
        "regressor"
    ].coef_

    coefficient_data[name] = coefficients

    regression_results.append({
        "Model": name,
        "Validation R2 Score": round(score, 4)
    })

    print(f"{name} R2 Score: {score:.4f}")


# Coefficient comparison
coefficient_table = pd.DataFrame(
    coefficient_data,
    index=regression_features
)

print("\nCOEFFICIENT COMPARISON")
print(coefficient_table.round(4))

# Count zero coefficients in Lasso
lasso_zero_count = np.sum(
    np.isclose(
        coefficient_table["Lasso"].values,
        0,
        atol=1e-8
    )
)

print("\nLasso Zero Coefficients:", lasso_zero_count)


# ==================================================
# 10. K-MEANS CUSTOMER SEGMENTATION
# ==================================================

print("\n========== K-MEANS CLUSTERING ==========")

cluster_features = [
    "income",
    "loan_amount",
    "credit_score"
]

cluster_data = df[cluster_features].copy()

cluster_scaler = StandardScaler()

scaled_data = cluster_scaler.fit_transform(cluster_data)

inertia_values = []
silhouette_values = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(scaled_data)

    inertia_values.append(kmeans.inertia_)

    score = silhouette_score(
        scaled_data,
        cluster_labels
    )

    silhouette_values.append(score)

    print(
        f"K={k} | "
        f"Inertia={kmeans.inertia_:.2f} | "
        f"Silhouette Score={score:.4f}"
    )


# ==================================================
# 11. ELBOW CURVE
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia_values,
    marker="o"
)

plt.title("K-Means Elbow Curve")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 9))
plt.grid(True)

plt.savefig("elbow_curve.png")
plt.close()

print("\nElbow curve saved successfully!")


# ==================================================
# 12. SILHOUETTE SCORE PLOT
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    silhouette_values,
    marker="o"
)

plt.title("K-Means Silhouette Scores")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 9))
plt.grid(True)

plt.savefig("silhouette_scores.png")
plt.close()

print("Silhouette plot saved successfully!")


# ==================================================
# 13. CUSTOMER SEGMENT DESCRIPTION
# ==================================================

best_k = list(range(2, 9))[
    np.argmax(silhouette_values)
]

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["customer_segment"] = final_kmeans.fit_predict(
    scaled_data
)

print("\nBest K:", best_k)

print("\nCUSTOMER SEGMENT DESCRIPTION")

segment_summary = df.groupby(
    "customer_segment"
)[cluster_features].mean().round(2)

print(segment_summary)

print("\nK-MEANS CLUSTERING COMPLETED!")


# ==================================================
# 14. CLASSIFICATION RESULTS TABLE
# ==================================================

print("\n========== FINAL MODEL RESULTS ==========")

classification_results = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Validation Score": round(logistic_accuracy, 4)
    },
    {
        "Model": "Decision Tree",
        "Validation Score": round(tree_accuracy, 4)
    },
    {
        "Model": "Random Forest",
        "Validation Score": round(forest_accuracy, 4)
    },
    {
        "Model": "XGBoost",
        "Validation Score": round(xgb_accuracy, 4)
    }
])

print("\nCLASSIFICATION RESULTS")
print(classification_results.to_string(index=False))

regression_results_df = pd.DataFrame(regression_results)

print("\nREGRESSION RESULTS")
print(regression_results_df.to_string(index=False))


# Save result tables
classification_results.to_csv(
    "classification_results.csv",
    index=False
)

regression_results_df.to_csv(
    "regression_results.csv",
    index=False
)

print("\nRESULT TABLES SAVED SUCCESSFULLY!")


# ==================================================
# 15. RANDOM FOREST TOP 5 FEATURES
# ==================================================

print("\n========== RANDOM FOREST TOP 5 FEATURES ==========")

forest_feature_names = (
    random_forest_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

forest_importance = (
    random_forest_model
    .named_steps["classifier"]
    .feature_importances_
)

forest_importance_df = pd.DataFrame({
    "Feature": forest_feature_names,
    "Importance": forest_importance
}).sort_values(
    "Importance",
    ascending=False
)

print(forest_importance_df.head(5).to_string(index=False))


# ==================================================
# 16. XGBOOST TOP 5 FEATURES
# ==================================================

print("\n========== XGBOOST TOP 5 FEATURES ==========")

xgb_feature_names = (
    xgb_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

xgb_importance = xgb_model.feature_importances_

xgb_importance_df = pd.DataFrame({
    "Feature": xgb_feature_names,
    "Importance": xgb_importance
}).sort_values(
    "Importance",
    ascending=False
)

print(xgb_importance_df.head(5).to_string(index=False))


# ==================================================
# TASK 3 COMPLETED
# ==================================================

print("\n==========================================")
print("TASK 3 EXECUTION COMPLETED SUCCESSFULLY!")
print("All models and clustering tasks completed.")