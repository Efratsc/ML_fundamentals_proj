import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to the path so you can import utils or config if needed
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# === Load processed train set ===
data_dir = os.path.abspath(os.path.join("..", "data", "processed"))
X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

# === Merge for analysis ===
df = pd.concat([X_train, y_train], axis=1)

# === Inspect types and missing values ===
print("Column types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDataFrame info:")
df.info()

# === Check class distribution ===
target_column = "class"
if target_column in df.columns:
    print("\nTarget class distribution:\n", df[target_column].value_counts())
    print(
        "\nTarget class proportions:\n",
        df[target_column].value_counts(normalize=True),
    )
else:
    print("\nERROR: Target column not found!")

# === Basic statistics ===
print("\nBasic statistics:\n", df.describe().T)

# === Visualize class distribution ===
plt.figure(figsize=(6, 4))
sns.countplot(x=target_column, data=df)
plt.title("Class Distribution")
plt.savefig(os.path.join("..", "results", "class_distribution.png"))
plt.show()

# === Correlation Matrix ===
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join("..", "results", "correlation_matrix.png"))
plt.show()

# === Save combined train set for future use ===
df.to_csv(os.path.join(data_dir, "train_combined.csv"), index=False)
print(f"\nCombined train set saved to: {os.path.join(data_dir, 'train_combined.csv')}")
