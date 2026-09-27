from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(..., examples=["Sneaker"])
    confidence: float = Field(..., ge=0, le=1, examples=[0.93])


class PredictResponse(BaseModel):
    prediction: Prediction
    top_k: list[Prediction] | None = None
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str