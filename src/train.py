import argparse
import json
import os
import time
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Import MLflow tracker
from mlflow_tracking import MLflowTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A comprehensive model training pipeline
    for ML projects with MLflow tracking.
    """

    def __init__(self, config_path: str = "config/train_config.yaml"):
        """
        Initialize the model trainer.

        Args:
            config_path: Path to the training configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.training_time = None
        self.metrics = {}
        self.mlflow_tracker = MLflowTracker()

    def load_config(self) -> dict:
        """Load training configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)

            if config is None:
                logger.warning(
                    f"Config file {self.config_path} is empty, using defaults"
                )
                return self.get_default_config()

            logger.info(f"Loaded configuration from {self.config_path}")
            return config

        except FileNotFoundError:
            logger.warning(
                f"Config file {self.config_path} not found, using defaults"
            )
            return self.get_default_config()
        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing YAML file {self.config_path}: {e}"
            )
            logger.warning("Using default configuration")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            logger.warning("Using default configuration")
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            "model": {
                "type": "RandomForestClassifier",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                },
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 5,
            },
            "paths": {
                "data_dir": "data/processed/",
                "model_dir": "models/",
                "metrics_dir": "results/",
                "logs_dir": "logs/",
            },
        }

    def load_data(self) -> tuple:
        """Load training and testing data."""
        logger.info("Loading training and testing data")

        data_dir = self.config["paths"]["data_dir"]

        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        y_train = pd.read_csv(
            os.path.join(data_dir, "y_train.csv")
        ).iloc[:, 0]
        y_test = pd.read_csv(
            os.path.join(data_dir, "y_test.csv")
        ).iloc[:, 0]

        logger.info(
            f"Loaded data - X_train: {X_train.shape}, X_test: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    def create_model(self) -> RandomForestClassifier:
        """Create and configure the model."""
        logger.info("Creating RandomForestClassifier model")

        model_params = self.config["model"]["params"]
        self.model = RandomForestClassifier(**model_params)

        logger.info(f"Model created with parameters: {model_params}")
        return self.model

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model and log training time."""
        logger.info("Starting model training")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        logger.info(
            f"Model training completed in {self.training_time:.2f} seconds"
        )

    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame = None,
    ) -> dict:
        """Evaluate the model and calculate metrics."""
        logger.info("Evaluating model performance")

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        train_size = len(X_train) if X_train is not None else None

        self.metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time": self.training_time,
            "test_size": len(X_test),
            "train_size": train_size,
        }

        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Model F1-score: {f1:.4f}")

        if train_size is not None:
            self.mlflow_tracker.log_metric("train_size", train_size)

        self.mlflow_tracker.log_hyperparameters(self.config["model"]["params"])
        print("Metrics before logging to MLflow:", self.metrics)

        self.mlflow_tracker.log_metrics(self.metrics)

        return self.metrics

    def cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict:
        """Perform cross-validation."""
        logger.info("Performing cross-validation")

        cv_folds = self.config["training"]["cv_folds"]
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=cv_folds,
            scoring="accuracy",
        )

        cv_metrics = {
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

        logger.info(
            f"Cross-validation accuracy: {cv_scores.mean():.4f} "
            f"(+/- {cv_scores.std() * 2:.4f})"
        )

        self.metrics.update(cv_metrics)

        return cv_metrics

    def plot_confusion_matrix(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: str = None,
    ) -> None:
        """Plot and save confusion matrix."""
        logger.info("Creating confusion matrix plot")

        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.model.classes_,
            yticklabels=self.model.classes_,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def save_model(self, model_path: str = None) -> None:
        """Save the trained model to disk."""
        if model_path is None:
            model_dir = self.config["paths"]["model_dir"]
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.pkl")

        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    def save_metrics(self, metrics_path: str = None) -> None:
        """Save training metrics to JSON file."""
        if metrics_path is None:
            metrics_dir = self.config["paths"]["metrics_dir"]
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "metrics.json")

        self.metrics["timestamp"] = datetime.now().isoformat()

        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

    def train_pipeline_with_mlflow(self) -> dict:
        """Complete training pipeline with MLflow tracking."""
        logger.info("Starting complete training pipeline with MLflow tracking")

        self.mlflow_tracker.start_run()

        try:
            X_train, X_test, y_train, y_test = self.load_data()

            self.mlflow_tracker.log_data_info(X_train, X_test, y_train, y_test)

            self.create_model()

            self.mlflow_tracker.log_hyperparameters(
                self.config["model"]["params"]
            )

            _ = self.cross_validate(X_train, y_train)

            self.train_model(X_train, y_train)

            self.evaluate_model(X_test, y_test, X_train=X_train)

            self.mlflow_tracker.log_metrics(self.metrics)

            self.mlflow_tracker.log_model(self.model)

            confusion_matrix_path = os.path.join(
                self.config["paths"]["metrics_dir"], "confusion_matrix.png"
            )
            self.mlflow_tracker.log_confusion_matrix(
                y_test, self.model.predict(X_test), confusion_matrix_path
            )

            feature_importance_path = os.path.join(
                self.config["paths"]["metrics_dir"], "feature_importance.png"
            )
            self.mlflow_tracker.log_feature_importance(
                self.model,
                X_train.columns,
                feature_importance_path,
            )

            self.mlflow_tracker.register_model()

            self.save_model()
            self.save_metrics()

            logger.info(
                "Training pipeline with MLflow tracking completed successfully"
            )
            return self.metrics

        finally:
            self.mlflow_tracker.end_run()

    def train_pipeline(self) -> dict:
        """Complete training pipeline
         (without MLflow for backward compatibility)."""
        logger.info("Starting complete training pipeline")

        X_train, X_test, y_train, y_test = self.load_data()

        self.create_model()

        _ = self.cross_validate(X_train, y_train)

        self.train_model(X_train, y_train)

        self.evaluate_model(X_test, y_test, X_train=X_train)

        confusion_matrix_path = os.path.join(
            self.config["paths"]["metrics_dir"], "confusion_matrix.png"
        )
        self.plot_confusion_matrix(X_test, y_test, confusion_matrix_path)

        self.save_model()
        self.save_metrics()

        logger.info("Training pipeline completed successfully")
        return self.metrics


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Path to save training metrics",
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Use MLflow tracking",
    )

    args = parser.parse_args()

    trainer = ModelTrainer(config_path=args.config)

    if args.use_mlflow:
        metrics = trainer.train_pipeline_with_mlflow()
    else:
        metrics = trainer.train_pipeline()

    print("\nTraining Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Cross-validation Accuracy: {metrics['cv_mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()
