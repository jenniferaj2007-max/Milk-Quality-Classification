import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load test data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
X_test_scaled = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Load models and their required test data
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

# Create confusion matrix plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, (model, features)) in zip(axes.flatten(), models.items()):
    predictions = model.predict(features)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)

plt.suptitle("Milk Quality Classification - Confusion Matrices")
plt.tight_layout()

# Save figure
output_path = RESULTS_DIR / "confusion_matrices.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Confusion matrices saved successfully!")
print("Saved at:", output_path)