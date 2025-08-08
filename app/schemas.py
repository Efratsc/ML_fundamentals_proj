# app/schemas.py

from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """
    Schema for incoming prediction requests.
    Expects a list of feature values (floats).
    """
    features: List[float] = Field(
        ...,
        description="List of numerical features for prediction",
        example=[0.1, 0.5, 3.2, 7.8]
    )


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    Returns the predicted class label and optionally the probabilities.
    """
    prediction: int = Field(..., description="Predicted class label")
    probabilities: List[float] = Field(
        None,
        description="Optional list of class probabilities",
        example=[0.1, 0.9]
    )
