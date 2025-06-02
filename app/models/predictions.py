from pydantic import BaseModel
from typing import List

class PredictInput(BaseModel):
    features: List[float]
    model: str

class PredictOutput(BaseModel):
    predicted_class: str
    model_used: str
    features: List[float]
    confidence: float
