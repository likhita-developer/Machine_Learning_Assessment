Task 3 – Models: Supervised and Unsupervised

Objective

Build and evaluate supervised and unsupervised machine learning models using a loan dataset.

Dataset

- 500 customer records
- Features include age, income, loan amount, credit score, employment years, gender, property area and education.
- Target variable: "loan_default"

1. Classification Models

Trained four classification models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier with early stopping

Classification Results

Model| Validation Accuracy
Logistic Regression| 0.67
Decision Tree| 0.67
Random Forest| 0.67
XGBoost| 0.72

Random Forest OOB Score: 0.645

XGBoost achieved the highest validation accuracy of 0.72.

2. Prediction Probabilities

Displayed predicted classes and probability scores for five validation records using Logistic Regression.

3. Feature Importance

Top five features were extracted from Random Forest and XGBoost.

Random Forest important features:

- Credit score
- Income
- Loan amount
- Age
- Employment years

XGBoost important features:

- Credit score
- Property area (Rural)
- Income
- Property area (Urban)
- Age

4. Regression Models

Used Linear Regression, Ridge and Lasso to predict loan amount.

Model| Validation R² Score
Linear Regression| -0.018
Ridge| -0.018
Lasso| -0.018

Lasso zero coefficients: 0.

The negative R² indicates that these selected features did not predict loan amount effectively on this generated dataset.

5. K-Means Clustering

Applied K-Means clustering using:

- Income
- Loan amount
- Credit score

Tested cluster values from K=2 to K=8.

- Best K based on silhouette score: 6
- Best silhouette score: 0.3004
- Generated elbow curve and silhouette score plots.
- Created customer segment summaries.

6. Output Files

- "main.py"
- "classification_results.csv"
- "regression_results.csv"
- "elbow_curve.png"
- "silhouette_scores.png"
- "README.md"

Conclusion

Successfully implemented classification, regression, feature importance analysis and customer segmentation using supervised and unsupervised machine learning techniques.