import os
import logging
from typing import Tuple, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    Preprocessing pipeline for text classification dataset:
    Features: tweet text only (no numeric scaling)
    Target: 'class' label encoded
    """

    def __init__(
        self,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.columns_to_drop = []

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip whitespace from column names.
        """
        df.columns = df.columns.str.strip()
        logger.info("Cleaned column names")
        return df

    def drop_irrelevant_columns(
        self, df: pd.DataFrame, columns_to_drop: List[str]
    ) -> pd.DataFrame:
        """
        Drop columns not needed for modeling.
        """
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_cols)
        logger.info(f"Dropped columns: {existing_cols}")
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label encode the target column.
        """
        df[self.target_column] = self.label_encoder.fit_transform(
            df[self.target_column]
        )
        logger.info(f"Encoded target column '{self.target_column}'")
        return df

    def create_features_labels_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Split dataset into features (tweet text) and labels.
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")

        # Features = only the tweet text column (assume column named "tweet")
        X = df["tweet"]
        y = df[self.target_column]
        logger.info(
            f"Features and labels split done. Feature shape: {X.shape}, Labels shape: {y.shape}"
        )
        return X, y

    def split_data(
        self, X: pd.Series, y: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split into train and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(f"Split data into train/test sets with test_size={self.test_size}")
        logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def save_split_datasets(
        self,
        X_train: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str = "../data/processed/",
    ) -> None:
        """
        Save split datasets to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False, header=True)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False, header=True)
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)

        # Save the label encoder to use later for decoding predictions
        artifacts_dir = os.path.join(output_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(
            self.label_encoder, os.path.join(artifacts_dir, "label_encoder.pkl")
        )
        logger.info(f"Saved label encoder at {artifacts_dir}")

        logger.info(f"Saved train/test splits at {output_dir}")

    def preprocess_pipeline(
        self, df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Run full preprocessing pipeline.
        """
        logger.info("Starting preprocessing pipeline")

        df = self.clean_column_names(df)

        if columns_to_drop:
            df = self.drop_irrelevant_columns(df, columns_to_drop)

        df = self.encode_target(df)

        X, y = self.create_features_labels_split(df)

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        logger.info("Preprocessing pipeline completed successfully")

        return X_train, X_test, y_train, y_test
