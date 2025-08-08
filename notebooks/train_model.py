import sys
import os
import mlflow

# Add src folder to Python path for importing ModelTrainer
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from train import ModelTrainer

# Base directory of the current notebook folder
base_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure necessary directories exist
os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "mlruns"), exist_ok=True)


def main():
    print("Starting model training pipeline with MLflow tracking...")

    # Absolute path to the config YAML
    config_path = os.path.abspath(os.path.join("..", "config", "train_config.yaml"))

    trainer = ModelTrainer(config_path=config_path)

    # Update paths in the config to absolute paths within the current notebook folder
    trainer.config["paths"]["data_dir"] = os.path.abspath(os.path.join("..", "data", "processed"))
    trainer.config["paths"]["model_dir"] = os.path.join(base_dir, "models")
    trainer.config["paths"]["metrics_dir"] = os.path.join(base_dir, "results")
    trainer.config["paths"]["logs_dir"] = os.path.join(base_dir, "logs")

    # Setup MLflow tracking URI and experiment name
    mlruns_path = os.path.join(base_dir, "mlruns")
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("domestic_violence_prediction")

    # Run training pipeline with MLflow tracking enabled
    _ = trainer.train_pipeline_with_mlflow()

    print("\nTraining completed!")
    print(f"Model saved to: {trainer.config['paths']['model_dir']}/model.pkl")
    print(f"Metrics saved to: {trainer.config['paths']['metrics_dir']}/metrics.json")
    print(f"Confusion matrix saved to: {trainer.config['paths']['metrics_dir']}/confusion_matrix.png")
    print("MLflow experiment: domestic_violence_prediction")

    tracking_uri_str = f"MLflow tracking URI: file://{mlruns_path}"
    if len(tracking_uri_str) > 79:
        print("MLflow tracking URI:")
        print(f"file://{mlruns_path}")
    else:
        print(tracking_uri_str)


if __name__ == "__main__":
    main()
