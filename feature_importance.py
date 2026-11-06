import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load Random Forest model
model = joblib.load(DATA_DIR / "random_forest_model.pkl")

# Load feature names
X_train = pd.read_csv(DATA_DIR / "X_train.csv")

# Extract feature importance
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nMILK QUALITY FEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

# Save feature importance
importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)

# Create visualization
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Milk Quality Classification - Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()

plt.tight_layout()

# Save graph
output_path = RESULTS_DIR / "feature_importance.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("\nFeature importance analysis saved successfully!")