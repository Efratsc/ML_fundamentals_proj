from preprocess import DataPreprocessor
from data_loader import load_raw_data
import sys
import os
import pandas as pd

# Add src to the path
sys.path.append(os.path.abspath(os.path.join("..", "src")))

# Create logs directory in the project root (not in notebooks)
project_root = os.path.abspath(os.path.join(".."))
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)


def main():
    # Load the raw data
    print("Loading raw data...")
    df = load_raw_data("Domestic violence.csv")
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Define target column
    target_column = "Violence "

    # Define columns to drop (if any)
    columns_to_drop = ["SL. No"]  # Drop the serial number column

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        target_column=target_column, test_size=0.2, random_state=42
    )

    # Run the complete preprocessing pipeline
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

    # Print some info about the target variable
    print(f"\nTarget variable '{target_column}' distribution:")
    print(y_train.value_counts())


if __name__ == "__main__":
    main()
