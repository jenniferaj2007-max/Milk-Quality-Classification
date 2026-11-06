import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

# Locate original dataset
dataset_files = [
    file for file in DATA_DIR.rglob("*")
    if file.is_file() and file.name.lower() == "milknew.csv"
]

if not dataset_files:
    raise FileNotFoundError("milknew.csv not found!")

# Load original dataset
df = pd.read_csv(dataset_files[0])

print("Original dataset shape:", df.shape)
print("Duplicate rows found:", df.duplicated().sum())
print("Missing values before cleaning:")
print(df.isnull().sum())

# Clean column names
df.columns = df.columns.str.strip()

# Remove rows only if they contain missing values
df = df.dropna().copy()

# Standardize grade labels
df["Grade"] = df["Grade"].astype(str).str.strip().str.lower()

# Save cleaned dataset without removing repeated observations
OUTPUT_PATH = DATA_DIR / "cleaned_milk_quality.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Remaining duplicate rows:", df.duplicated().sum())

print("\nGrade distribution:")
print(df["Grade"].value_counts())

print("\nCleaned dataset saved successfully!")