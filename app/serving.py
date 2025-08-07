import joblib
import os
from fastapi import HTTPException


class MultiModelAPI:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        """
        Load all models from the model directory.
        """
        for file in os.listdir(self.model_dir):
            if file.endswith(".pkl"):
                model_name = file.split(".pkl")[0]
                with open(os.path.join(self.model_dir, file), "rb") as f:
                    self.models[model_name] = joblib.load(f)

    def predict(self, model_name: str, features: list):
        """
        Make predictions using the specified model.
        """
        model = self.models.get(model_name)
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        prediction = model.predict([features])
        return prediction[0]
