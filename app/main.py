from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from app.serving import MultiModelAPI

app = FastAPI()

# Initialize the multi-model loader
model_api = MultiModelAPI(model_dir="models")

# Define input data schema


class PredictionRequest(BaseModel):
    model_name: str
    features: List[float]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Model ML API"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        prediction = model_api.predict(request.model_name, request.features)
        return {"model": request.model_name, "prediction": prediction}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
