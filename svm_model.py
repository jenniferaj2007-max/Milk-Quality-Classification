import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load scaled training and testing data
X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("=" * 55)
print("       SUPPORT VECTOR MACHINE MODEL")
print("=" * 55)

# Train SVM model
model = SVC(
    kernel="rbf",
    C=1.0,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        zero_division=0
    )
)

# Save trained model
joblib.dump(model, DATA_DIR / "svm_model.pkl")

# Save results
results = pd.DataFrame([{
    "Model": "SVM",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}])

results.to_csv(
    RESULTS_DIR / "svm_results.csv",
    index=False
)

print("\nSVM model saved successfully!")
print("SVM results saved successfully!")