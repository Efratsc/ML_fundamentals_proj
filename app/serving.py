import joblib
import os
from fastapi import HTTPException
from typing import List

class MultiModelAPI:


    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        """
        Load all models from the model directory.
        """
        for file in os.listdir(self.model_dir):
            if file.endswith(".pkl"):
                model_name = file[:-4]  # safer than split
                try:
                    with open(os.path.join(self.model_dir, file), "rb") as f:
                        self.models[model_name] = joblib.load(f)
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")

    def predict(self, model_name: str, features: List[float]):
        """
        Make predictions using the specified model.
        :param model_name: The key name of the loaded model
        :param features: List of feature values for prediction
        :return: Prediction result (typically a class label or value)
        """
        model = self.models.get(model_name)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )

        try:
            prediction = model.predict([features])
            return prediction[0]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error making prediction: {e}"
            )
