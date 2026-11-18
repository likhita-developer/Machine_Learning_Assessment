import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

import joblib


# ==========================================
# 1. GENERATE LOAN DATASET
# ==========================================

np.random.seed(42)

n = 500

df = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 150000, n),
    "loan_amount": np.random.randint(10000, 200000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 30, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "property_area": np.random.choice(
        ["Urban", "Rural", "Semiurban"], n
    ),
    "education": np.random.choice(
        ["Graduate", "Postgraduate", "Undergraduate"], n
    ),
    "loan_date": pd.date_range(
        start="2024-01-01", periods=n, freq="D"
    )
})

# Create target variable

default_probability = (
    0.15
    + 0.20 * (df["credit_score"] < 550)
    + 0.15 * (df["income"] < 40000)
    + 0.10 * (df["loan_amount"] > 150000)
)

df["loan_default"] = (
    np.random.rand(n) < default_probability
).astype(int)


# ==========================================
# 2. INTRODUCE MISSING VALUES
# ==========================================

for column in ["age", "income", "credit_score"]:
    df.loc[
        df.sample(frac=0.05, random_state=42).index,
        column
    ] = np.nan

df.loc[
    df.sample(frac=0.05, random_state=10).index,
    "gender"
] = np.nan

df.loc[
    df.sample(frac=0.05, random_state=20).index,
    "property_area"
] = np.nan


print("\nLOAN DATASET")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["loan_default"].value_counts())


# ==========================================
# 3. SIMPLE IMPUTER
# ==========================================

print("\nSIMPLE IMPUTER")

numeric_cols = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "employment_years"
]

categorical_cols = [
    "gender",
    "property_area",
    "education"
]

numeric_imputer = SimpleImputer(strategy="median")

numeric_data = pd.DataFrame(
    numeric_imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

categorical_data = pd.DataFrame(
    categorical_imputer.fit_transform(df[categorical_cols]),
    columns=categorical_cols
)

print("\nNumeric missing values after imputation:")
print(numeric_data.isnull().sum())

print("\nCategorical missing values after imputation:")
print(categorical_data.isnull().sum())


# ==========================================
# 4. KNN IMPUTER
# ==========================================

print("\nKNN IMPUTER")

knn_imputer = KNNImputer(n_neighbors=5)

knn_data = pd.DataFrame(
    knn_imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)

print("\nMissing values after KNN imputation:")
print(knn_data.isnull().sum())

print("\nImputation completed successfully.")


# ==========================================
# 5. CATEGORICAL ENCODING
# ==========================================

print("\nCATEGORICAL ENCODING")

onehot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

onehot_data = onehot_encoder.fit_transform(
    categorical_data[["gender", "property_area"]]
)

onehot_columns = onehot_encoder.get_feature_names_out(
    ["gender", "property_area"]
)

onehot_df = pd.DataFrame(
    onehot_data,
    columns=onehot_columns
)

print("\nOneHot Encoded Data:")
print(onehot_df.head())


ordinal_encoder = OrdinalEncoder(
    categories=[[
        "Undergraduate",
        "Graduate",
        "Postgraduate"
    ]]
)

ordinal_data = ordinal_encoder.fit_transform(
    categorical_data[["education"]]
)

ordinal_df = pd.DataFrame(
    ordinal_data,
    columns=["education_encoded"]
)

print("\nOrdinal Encoded Education:")
print(ordinal_df.head())

print("\nEncoding completed successfully.")


# ==========================================
# 6. FEATURE ENGINEERING DEMONSTRATION
# ==========================================

print("\nFEATURE ENGINEERING")

feature_df = df.copy()

feature_df["loan_year"] = feature_df["loan_date"].dt.year
feature_df["loan_month"] = feature_df["loan_date"].dt.month
feature_df["loan_day"] = feature_df["loan_date"].dt.day

feature_df["income_loan_ratio"] = (
    feature_df["income"] / feature_df["loan_amount"]
)

feature_df["log_income"] = np.log1p(
    feature_df["income"]
)

print("\nNew Features Created:")

print(feature_df[
    [
        "loan_year",
        "loan_month",
        "loan_day",
        "income_loan_ratio",
        "log_income"
    ]
].head())

print("\nUpdated Dataset Shape:", feature_df.shape)

print("\nFeature engineering completed successfully.")


# ==========================================
# 7. CUSTOM FEATURE ENGINEERING TRANSFORMER
# ==========================================

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["loan_date"] = pd.to_datetime(X["loan_date"])

        X["loan_year"] = X["loan_date"].dt.year
        X["loan_month"] = X["loan_date"].dt.month
        X["loan_day"] = X["loan_date"].dt.day

        X["income_loan_ratio"] = (
            X["income"] / X["loan_amount"]
        )

        X["log_income"] = np.log1p(X["income"])

        X = X.drop(columns=["loan_date"])

        return X


# ==========================================
# 8. PREPROCESSING PIPELINES
# ==========================================

print("\nMACHINE LEARNING PIPELINE")

X = df.drop(columns=["loan_default"])
y = df["loan_default"]

numeric_features = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "loan_year",
    "loan_month",
    "loan_day",
    "income_loan_ratio",
    "log_income"
]

categorical_features = [
    "gender",
    "property_area"
]

ordinal_features = ["education"]


# Numerical pipeline

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical pipeline

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Ordinal pipeline

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[[
            "Undergraduate",
            "Graduate",
            "Postgraduate"
        ]]
    ))
])


# ==========================================
# 9. COLUMN TRANSFORMER
# ==========================================

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
    ("ord", ordinal_pipeline, ordinal_features)
])


# ==========================================
# 10. COMPLETE MACHINE LEARNING PIPELINE
# ==========================================

model_pipeline = Pipeline([
    ("feature_engineering", LoanFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ==========================================
# 11. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 12. CROSS VALIDATION
# ==========================================

cv_scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:", cv_scores)

print("Mean Accuracy:", round(cv_scores.mean(), 4))

print("Standard Deviation:", round(cv_scores.std(), 4))


# ==========================================
# 13. TRAIN FINAL MODEL
# ==========================================

model_pipeline.fit(X_train, y_train)

print("\nPipeline training completed successfully!")


# ==========================================
# 14. SAVE PIPELINE USING JOBLIB
# ==========================================

print("\nSAVING FINAL ML PIPELINE")

joblib.dump(model_pipeline, "loan_pipeline.pkl")

print("Final pipeline saved successfully!")


# ==========================================
# 15. LOAD SAVED PIPELINE
# ==========================================

loaded_pipeline = joblib.load("loan_pipeline.pkl")

print("Pipeline loaded successfully!")


# ==========================================
# 16. PREDICT FOR 3 NEW CUSTOMERS
# ==========================================

new_customers = pd.DataFrame({

    "age": [25, 40, 55],

    "income": [35000, 85000, 120000],

    "loan_amount": [150000, 80000, 50000],

    "credit_score": [450, 700, 800],

    "employment_years": [2, 12, 25],

    "gender": ["Male", "Female", "Male"],

    "property_area": ["Rural", "Urban", "Semiurban"],

    "education": [
        "Undergraduate",
        "Graduate",
        "Postgraduate"
    ],

    "loan_date": [
        "2025-01-10",
        "2025-02-15",
        "2025-03-20"
    ]
})


predictions = loaded_pipeline.predict(new_customers)

print("\nPREDICTIONS FOR 3 NEW CUSTOMERS")

print("Predictions:", predictions)

print("\nTASK 2 EXECUTION COMPLETED SUCCESSFULLY!")