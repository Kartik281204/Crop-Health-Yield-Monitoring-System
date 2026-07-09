from typing import List
from pydantic import BaseModel, Field


class ClassConfidence(BaseModel):
    label: str
    confidence: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1)
    top_3: List[ClassConfidence]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int


class ClassesResponse(BaseModel):
    classes: List[str]
