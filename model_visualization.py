import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load model comparison results
file_path = RESULTS_DIR / "model_performance_comparison.csv"

df = pd.read_csv(file_path)

# Plot accuracy comparison
plt.figure(figsize=(10, 6))

plt.bar(df["Model"], df["Accuracy"])

plt.title("Milk Quality Classification - Model Accuracy Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# Save graph
output_path = RESULTS_DIR / "model_performance_comparison.png"

plt.savefig(output_path, dpi=300)
plt.show()

print("Model visualization saved successfully!")
print("Graph location:", output_path)