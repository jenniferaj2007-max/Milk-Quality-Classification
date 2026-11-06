import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("       MILK QUALITY MODEL COMPARISON")
print("=" * 60)

# Load test datasets
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
X_test_scaled = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Load trained models
models = {
    "Logistic Regression": (
        joblib.load(DATA_DIR / "logistic_regression_model.pkl"),
        X_test_scaled
    ),
    "Decision Tree": (
        joblib.load(DATA_DIR / "decision_tree_model.pkl"),
        X_test
    ),
    "Random Forest": (
        joblib.load(DATA_DIR / "random_forest_model.pkl"),
        X_test
    ),
    "SVM": (
        joblib.load(DATA_DIR / "svm_model.pkl"),
        X_test_scaled
    )
}

# Evaluate all models
comparison_results = []

for model_name, (model, test_features) in models.items():

    predictions = model.predict(test_features)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    comparison_results.append({
        "Model": model_name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

# Create comparison table
comparison_df = pd.DataFrame(comparison_results)

print("\nMODEL PERFORMANCE COMPARISON\n")
print(comparison_df.to_string(index=False))

# Identify best model by accuracy
best_model = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

print("\nBest Performing Model:", best_model["Model"])
print("Best Accuracy:", best_model["Accuracy"], "%")

# Save comparison results
comparison_df.to_csv(
    RESULTS_DIR / "model_performance_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")