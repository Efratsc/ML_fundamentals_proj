import os
import logging
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib


# Set up logging with absolute path
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "logs"
)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "preprocessing.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessing pipeline for ML projects.
    """

    def __init__(
        self,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the preprocessor.

        Args:
            target_column: Name of the target variable
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="mean")
        self.columns_to_drop = []

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "mean"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Strategy for imputation ('mean', 'median',
                      'most_frequent', 'constant')

        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using {strategy} strategy")

        # Get numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Handle numerical columns
        if len(numerical_cols) > 0:
            if strategy in ["mean", "median"]:
                imputer = SimpleImputer(strategy=strategy)
                df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                logger.info(
                    f"Imputed {len(numerical_cols)} numerical columns "
                    f"using {strategy} strategy"
                )

        # Handle categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(
                        df[col].mode()[0]
                        if len(df[col].mode()) > 0
                        else "Unknown"
                    )
                    logger.info(
                        f"Imputed missing values in categorical column: {col}"
                    )

        null_count = df.isnull().sum().sum()
        logger.info(f"Missing values handled. Remaining nulls: {null_count}")
        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.

        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode. If None,
                     encodes all object columns.

        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features")

        if columns is None:
            columns = df.select_dtypes(include=["object"]).columns.tolist()

        df_encoded = df.copy()

        for col in columns:
            if col in df.columns and col != self.target_column:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded categorical column: {col}")

        logger.info(f"Encoded {len(columns)} categorical features")
        return df_encoded

    def normalize_numerical_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.

        Args:
            df: Input DataFrame
            columns: List of numerical columns to normalize. If None,
                     normalizes all numerical columns.

        Returns:
            DataFrame with normalized numerical features
        """
        logger.info("Normalizing numerical features")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target column from normalization
            if self.target_column in columns:
                columns.remove(self.target_column)

        df_normalized = df.copy()

        if len(columns) > 0:
            df_normalized[columns] = self.scaler.fit_transform(
                df_normalized[columns]
            )
            logger.info(f"Normalized {len(columns)} numerical features")

        return df_normalized

    def drop_irrelevant_columns(
        self, df: pd.DataFrame, columns_to_drop: List[str]
    ) -> pd.DataFrame:
        """
        Drop irrelevant columns from the dataset.

        Args:
            df: Input DataFrame
            columns_to_drop: List of column names to drop

        Returns:
            DataFrame with irrelevant columns removed
        """
        logger.info(f"Dropping {len(columns_to_drop)} irrelevant columns")

        # Only drop columns that exist in the DataFrame
        existing_columns = [
            col for col in columns_to_drop if col in df.columns
        ]
        df_cleaned = df.drop(columns=existing_columns)

        self.columns_to_drop = existing_columns
        logger.info(f"Dropped columns: {existing_columns}")

        return df_cleaned

    def create_features_labels_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features (X) and labels (y).

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features, labels)
        """
        logger.info("Creating features and labels split")

        if self.target_column not in df.columns:
            msg = (
                f"Target column '{self.target_column}' "
                "not found in DataFrame"
            )
            raise ValueError(msg)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Args:
            X: Features DataFrame
            y: Labels Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={self.test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            f"Training set shape: {X_train.shape}, "
            f"Testing set shape: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    def save_split_datasets(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str = "../data/processed/",
    ) -> None:
        """
        Save split datasets to disk.

        Args:
            X_train, X_test, y_train, y_test: Split datasets
            output_dir: Directory to save the datasets
        """
        logger.info(f"Saving split datasets to {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save datasets
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        # Save preprocessing artifacts
        artifacts_dir = os.path.join(output_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        joblib.dump(self.scaler, os.path.join(artifacts_dir, "scaler.pkl"))
        joblib.dump(
            self.label_encoders,
            os.path.join(artifacts_dir, "label_encoders.pkl"),
        )

        logger.info("Split datasets saved successfully")

    def preprocess_pipeline(
        self, df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            columns_to_drop: List of columns to drop

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting preprocessing pipeline")

        # Step 1: Handle missing values
        df = self.handle_missing_values(df)

        # Step 2: Drop irrelevant columns
        if columns_to_drop:
            df = self.drop_irrelevant_columns(df, columns_to_drop)

        # Step 3: Encode categorical features
        df = self.encode_categorical_features(df)

        # Step 4: Normalize numerical features
        df = self.normalize_numerical_features(df)

        # Step 5: Create features and labels split
        X, y = self.create_features_labels_split(df)

        # Step 6: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        logger.info("Preprocessing pipeline completed successfully")
        return X_train, X_test, y_train, y_test
