import pandas as pd
from pathlib import Path

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

# Search for milknew.csv inside all subfolders
dataset_files = [
    file for file in DATA_DIR.rglob("*")
    if file.is_file() and file.name.lower() == "milknew.csv"
]

if not dataset_files:
    print("ERROR: milknew.csv was not found!")
    print("Files and folders available inside Data:")

    for item in DATA_DIR.rglob("*"):
        print(item)

else:
    DATA_PATH = dataset_files[0]

    print("Dataset found at:", DATA_PATH)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 50)
    print("MILK QUALITY CLASSIFICATION SYSTEM")
    print("=" * 50)

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset information:")
    df.info()

    print("\nDataset loaded successfully!")