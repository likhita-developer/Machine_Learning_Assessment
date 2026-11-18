Task 1 – Concepts, Splits and Leakage

1. Train, Validation and Test Split

The dataset was divided into:

- Training: 60%
- Validation: 20%
- Testing: 20%

Used "random_state=42" and stratified splitting to maintain target class proportions.

2. Decision Tree Overfitting

Trained DecisionTreeClassifier models with max_depth values of 1, 3, 5, 10 and None.

Compared training and validation accuracy using a line plot to understand overfitting.

3. Logistic Regression Cross-Validation

Performed 5-fold cross-validation using Logistic Regression.

Calculated the mean accuracy and standard deviation to evaluate model consistency.

4. Data Leakage Demonstration

- Scaling before splitting can cause data leakage because information from validation and test sets influences preprocessing.
- This can result in misleading model evaluation.
- The correct approach is to split the data before scaling.
- The scaler must be fitted only on training data and used to transform validation and test data.

5. Supervised and Unsupervised Learning

- Spam Detection: Supervised Learning
- Customer Segmentation: Unsupervised Learning
- House Price Prediction: Supervised Learning
- Anomaly Detection: Usually Unsupervised Learning

6. Machine Learning Workflow

1. Define the problem.
2. Collect the dataset.
3. Explore and understand the data.
4. Clean and preprocess the data.
5. Split the dataset.
6. Train and evaluate the model.
7. Tune and deploy the best model.