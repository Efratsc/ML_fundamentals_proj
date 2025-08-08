import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from preprocess import DataPreprocessor
from data_loader import load_raw_data


# Add src to the path
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Create logs directory in the project root
project_root = os.path.abspath(os.path.join(".."))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

def main():
    # Load the raw data
    print("Loading raw data...")
    df = load_raw_data("labeled_data.csv")  # ✅ Updated filename
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Define target column
    target_column = "class"  # ✅ The label (0, 1, 2)

    # Optional: Filter only abuse-related labels (1 = offensive, 2 = hate speech)
    df = df[df["class"] != 0]  # You can adjust this as needed

    # Define columns to drop (keep only "tweet" and "class")
    columns_to_drop = ["count", "hate_speech", "offensive_language", "neither"]  # ✅ Drop unneeded columns

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        target_column=target_column,
        test_size=0.2,
        random_state=42
    )

    # Run the preprocessing pipeline
    print("Running preprocessing pipeline...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, columns_to_drop=columns_to_drop
    )

    # Save the split datasets
    print("Saving split datasets...")
    preprocessor.save_split_datasets(X_train, X_test, y_train, y_test)

    print("Preprocessing completed successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")

    # Show class distribution
    print(f"\nTarget variable '{target_column}' distribution:")
    print(y_train.value_counts())

if __name__ == "__main__":
    main()
