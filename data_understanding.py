import pandas as pd
from pathlib import Path

# Locate the project and dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

dataset_files = [
    file for file in DATA_DIR.rglob("*")
    if file.is_file() and file.name.lower() == "milknew.csv"
]

if not dataset_files:
    raise FileNotFoundError("milknew.csv not found inside Data folder!")

# Load dataset
df = pd.read_csv(dataset_files[0])

print("=" * 60)
print("          MILK QUALITY DATA UNDERSTANDING")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. Dataset Shape:")
print(df.shape)

# 2. First five records
print("\n2. First Five Records:")
print(df.head())

# 3. Last five records
print("\n3. Last Five Records:")
print(df.tail())

# 4. Column names
print("\n4. Column Names:")
print(df.columns.tolist())

# 5. Data types and non-null values
print("\n5. Dataset Information:")
df.info()

# 6. Statistical summary
print("\n6. Statistical Summary:")
print(df.describe(include="all"))

# 7. Missing values
print("\n7. Missing Values:")
print(df.isnull().sum())

# 8. Duplicate records
print("\n8. Duplicate Records:")
print(df.duplicated().sum())

# 9. Target distribution
if "Grade" in df.columns:
    print("\n9. Milk Quality Grade Distribution:")
    print(df["Grade"].value_counts())

    print("\nGrade Percentages:")
    print(df["Grade"].value_counts(normalize=True) * 100)

print("\nData understanding completed successfully!")