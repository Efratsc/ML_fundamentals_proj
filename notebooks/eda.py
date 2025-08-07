from data_loader import load_raw_data
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to the path so you can import data_loader
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Load your data
df = load_raw_data("Domestic violence.csv")

# Inspect column types and missing values
print("Column types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDataFrame info:")
df.info()

# Check target class distribution
print("\nColumns:", df.columns)
target_column = "target"  # Change this to your actual target column name
if target_column in df.columns:
    print("\nTarget class distribution:\n", df[target_column].value_counts())
    print(
        "\nTarget class proportions:\n", df[target_column].value_counts(normalize=True)
    )
else:
    print("\nPlease set 'target_column' to your actual target column name.")

# Generate basic data stats
print("\nBasic statistics:\n", df.describe().T)

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("../results/correlation_matrix.png")  # Save the plot
plt.show()

# Save cleaned data (example: drop rows with missing values)
df_clean = df.dropna()
processed_path = os.path.abspath(
    os.path.join("..", "data", "processed", "Domestic violence cleaned.csv")
)
df_clean.to_csv(processed_path, index=False)
print(f"\nCleaned data saved to {processed_path}")
