Task 2 – Preprocessing, Features and Pipeline

Objective

To perform data preprocessing, feature engineering, and build a complete machine learning pipeline for loan default prediction.

Dataset

- 500 customer records
- Numerical and categorical features
- Target variable: loan_default

1. Missing Value Handling

- SimpleImputer with median strategy for numerical columns.
- SimpleImputer with most frequent strategy for categorical columns.
- KNNImputer with 5 neighbors for numerical columns.

2. Categorical Encoding

- OneHotEncoder for gender and property_area.
- OrdinalEncoder for education.

3. Feature Engineering

Created the following features:

- loan_year
- loan_month
- loan_day
- income_loan_ratio
- log_income

4. Machine Learning Pipeline

Used:

- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder
- OrdinalEncoder
- LogisticRegression

Feature engineering and preprocessing are integrated into the pipeline to prevent preprocessing leakage during cross-validation.

5. Model Evaluation

- Cross-validation: 5 folds
- Mean accuracy: 71.75%
- Standard deviation: 3.02%

6. Model Saving and Prediction

- Saved the fitted pipeline using joblib.
- Loaded the saved pipeline successfully.
- Generated predictions for 3 new raw customer records.

Conclusion

Successfully implemented preprocessing, feature engineering, cross-validation, pipeline creation, model serialization, and loan default prediction.