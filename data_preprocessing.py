import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

DATA_PATH = DATA_DIR / "cleaned_milk_quality.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run data_cleaning.py first."
    )

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("       MILK QUALITY DATA PREPROCESSING")
print("=" * 55)

# Separate features and target
X = df.drop(columns=["Grade"])
y = df["Grade"].astype(str).str.strip().str.lower()

# Encode target labels
grade_mapping = {
    "low": 0,
    "medium": 1,
    "high": 2
}

y = y.map(grade_mapping)

if y.isnull().any():
    raise ValueError("Unexpected grade labels found in dataset.")

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors="raise")

print("\nOriginal dataset shape:", df.shape)
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Group identical complete records to prevent duplicate leakage
groups = pd.factorize(
    X.astype(str).agg("|".join, axis=1)
)[0]

# Stratified group splitting
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

# Scale features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns,
    index=X_test.index
)

# Save training and testing datasets
X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)

X_train_scaled.to_csv(
    DATA_DIR / "X_train_scaled.csv", index=False
)

X_test_scaled.to_csv(
    DATA_DIR / "X_test_scaled.csv", index=False
)

y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

# Save scaler and grade mapping
joblib.dump(scaler, DATA_DIR / "scaler.pkl")
joblib.dump(grade_mapping, DATA_DIR / "grade_mapping.pkl")

# Verify duplicate separation
train_groups = set(groups[train_indices])
test_groups = set(groups[test_indices])

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)

print(
    "\nDuplicate groups shared between training and testing:",
    len(train_groups.intersection(test_groups))
)

print("\nTraining grade distribution:")
print(y_train.value_counts())

print("\nTesting grade distribution:")
print(y_test.value_counts())

print("\nAll preprocessing files saved successfully!")
print("Data preprocessing completed!")