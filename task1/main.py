import pandas as pd
import numpy as np

# Create sample loan dataset
np.random.seed(42)

n = 500

data = {
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 150000, n),
    "loan_amount": np.random.randint(10000, 200000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 30, n),
    "loan_default": np.random.randint(0, 2, n)
}

df = pd.DataFrame(data)

# Display dataset
print("LOAN DATASET")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop("loan_default", axis=1)
y = df["loan_default"]

# First split: 60% training and 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Second split: 20% validation and 20% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Display split sizes
print("\nDATA SPLIT RESULTS")
print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Different tree depths
depths = [1, 3, 5, 10, None]

train_accuracies = []
val_accuracies = []

print("\nDECISION TREE OVERFITTING")

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"Depth {depth}: "
        f"Train Accuracy = {train_acc:.2f}, "
        f"Validation Accuracy = {val_acc:.2f}"
    )

# Plot accuracy comparison
plt.figure(figsize=(9, 5))

labels = ["1", "3", "5", "10", "None"]

plt.plot(labels, train_accuracies, marker="o",
         label="Training Accuracy")

plt.plot(labels, val_accuracies, marker="o",
         label="Validation Accuracy")

plt.xlabel("Decision Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting Analysis")
plt.legend()
plt.grid(True)

plt.savefig("decision_tree_overfitting.png")
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create Logistic Regression pipeline
log_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Perform 5-fold cross-validation on training data
cv_scores = cross_val_score(
    log_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nLOGISTIC REGRESSION CROSS-VALIDATION")
print("Fold Scores:", cv_scores)
print(f"Mean Accuracy: {cv_scores.mean():.2f}")
print(f"Standard Deviation: {cv_scores.std():.2f}")
print(f"CV Result: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")
from sklearn.preprocessing import StandardScaler

print("\nDATA LEAKAGE DEMONSTRATION")

# Incorrect approach: Scaling before splitting
scaler_wrong = StandardScaler()
X_scaled_wrong = scaler_wrong.fit_transform(X)

print("Incorrect Approach: Scaling before splitting")
print("Scaled dataset shape:", X_scaled_wrong.shape)

# Correct approach: Fit scaler only on training data
scaler_correct = StandardScaler()

X_train_scaled = scaler_correct.fit_transform(X_train)
X_val_scaled = scaler_correct.transform(X_val)
X_test_scaled = scaler_correct.transform(X_test)

print("\nCorrect Approach: Scaling after splitting")
print("Training shape:", X_train_scaled.shape)
print("Validation shape:", X_val_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

print("\nData leakage demonstration completed.")
print("\nSUPERVISED VS UNSUPERVISED LEARNING")

problems = {
    "Spam Detection": "Supervised Learning",
    "Customer Segmentation": "Unsupervised Learning",
    "House Price Prediction": "Supervised Learning",
    "Anomaly Detection": "Unsupervised Learning"
}

for problem, category in problems.items():
    print(f"{problem}: {category}")


print("\n7-STEP MACHINE LEARNING WORKFLOW")

workflow = [
    "1. Define the problem",
    "2. Collect the dataset",
    "3. Explore and understand the data",
    "4. Clean and preprocess the data",
    "5. Split the dataset",
    "6. Train and evaluate the model",
    "7. Tune and deploy the best model"
]

for step in workflow:
    print(step)