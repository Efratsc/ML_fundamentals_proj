# app/schemas.py

from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    features: List[float]
