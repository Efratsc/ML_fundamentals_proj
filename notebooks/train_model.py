from train import ModelTrainer
import sys
import os
import mlflow

# Add src to the path
base_dir = os.path.dirname(os.path.abspath(__file__))  # /full/path/to/ml_project

os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "mlruns"), exist_ok=True)

# Add the src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def main():
    print("Starting model training pipeline with MLflow tracking...")

    # Set absolute path to config file
    config_path = os.path.abspath(os.path.join("..", "config", "train_config.yaml"))
    trainer = ModelTrainer(config_path=config_path)

    # Update the paths in the config to absolute paths
    trainer.config["paths"]["data_dir"] = os.path.abspath(
        os.path.join("..", "data", "processed")
    )
    trainer.config["paths"]["model_dir"] = os.path.join(base_dir, "models")
    trainer.config["paths"]["metrics_dir"] = os.path.join(base_dir, "results")
    trainer.config["paths"]["logs_dir"] = os.path.join(base_dir, "logs")

    # Set MLflow tracking URI and experiment
    mlruns_path = os.path.join(base_dir, "mlruns")
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("domestic_violence_prediction")

    # Run the training pipeline with MLflow tracking
    metrics = trainer.train_pipeline_with_mlflow()

    # Print summary
    print("\nTraining completed!")
    print(f"Model saved to: models/model.pkl")
    print(f"Metrics saved to: results/metrics.json")
    print(f"Confusion matrix saved to: results/confusion_matrix.png")
    print(f"MLflow experiment: domestic_violence_prediction")
    print(f"MLflow tracking URI: file://{mlruns_path}")


if __name__ == "__main__":
    main()
