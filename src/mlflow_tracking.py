import mlflow
import mlflow.sklearn
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Helper class for MLflow experiment tracking.
    """

    def __init__(self, experiment_name: str = "domestic_violence_prediction"):
        self.experiment_name = experiment_name
        self.run = None
        self.setup_mlflow()

    def setup_mlflow(self):
        """Set MLflow tracking URI and experiment."""
        mlruns_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "mlruns")
        )
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow experiment set to: {self.experiment_name}")

    def start_run(self, run_name: str = None):
        """Start a new MLflow run, ending any active run first."""
        if mlflow.active_run() is not None:
            logger.warning(
                "An MLflow run is already active. Ending it before starting a new run."
            )
            mlflow.end_run()
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")

    def log_hyperparameters(self, params: dict):
        """Log hyperparameters to MLflow."""
        mlflow.log_params(params)
        logger.info(f"Logged hyperparameters: {params}")

    def log_metrics(self, metrics):
        """Log metrics to MLflow, filtering out None values."""
        filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
        print("Filtered metrics for MLflow:", filtered_metrics)  # Optional debug print
        mlflow.log_metrics(filtered_metrics)
        logger.info(f"Logged metrics: {filtered_metrics}")

    def log_model(self, model, model_name: str = "random_forest_model"):
        """Log a sklearn model to MLflow."""
        mlflow.sklearn.log_model(model, model_name)
        logger.info(f"Logged model: {model_name}")

    def log_confusion_matrix(
        self, y_true, y_pred, save_path: str = "confusion_matrix.png"
    ):
        """Generate, save, and log confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(save_path)
        logger.info(f"Logged confusion matrix: {save_path}")

    def log_feature_importance(
        self, model, feature_names, save_path: str = "feature_importance.png"
    ):
        """Generate, save, and log feature importance plot."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=feature_importance_df.head(10),
                x="importance",
                y="feature",
            )
            plt.title("Top 10 Feature Importances")
            plt.xlabel("Importance")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(save_path)
            logger.info(f"Logged feature importance: {save_path}")

    def log_data_info(self, X_train, X_test, y_train, y_test):
        """Log dataset information as a JSON artifact."""
        data_info = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "n_classes": len(y_train.unique()),
            "class_distribution_train": y_train.value_counts().to_dict(),
            "class_distribution_test": y_test.value_counts().to_dict(),
        }
        mlflow.log_dict(data_info, "data_info.json")
        logger.info(f"Logged data info: {data_info}")

    def end_run(self):
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("Ended MLflow run")

    def register_model(
        self, model_name: str = "domestic_violence_model", version: str = "v1.0"
    ):
        """Register the model in the MLflow Model Registry."""
        try:
            if self.run is None:
                raise RuntimeError("No active run to register model from")
            model_uri = f"runs:/{self.run.info.run_id}/random_forest_model"
            mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not register model: {e}")

    def get_experiment_info(self):
        """Return basic experiment info including total runs."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return {
                "experiment_name": self.experiment_name,
                "experiment_id": experiment.experiment_id,
                "total_runs": len(runs),
            }
        return None
