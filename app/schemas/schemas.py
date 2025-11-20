from pydantic import BaseModel
from typing import Any


class AvailableModelsResponse(BaseModel):
    available_models: dict[str, str]


class TrainModelRequest(BaseModel):
    model_name: str
    hyperparameters: dict[str, Any] = {}
    features: list[list[float]]
    target: list[int]


class TrainModelResponse(BaseModel):
    message: str
    trained_model_id: str


class TrainedModelInfo(BaseModel):
    id: str
    model_name: str
    hyperparameters: dict[str, Any]
    model_path: str


class ModelPredictRequest(BaseModel):
    features: list[list[float]]


class ModelPredictResponse(BaseModel):
    model_id: str
    preds: list[int]


class RetrainModelRequest(BaseModel):
    hyperparameters: dict[str, Any] = {}
    features: list[list[float]]
    target: list[int]


class UserCreate(BaseModel):
    username: str
    password: str


class User(BaseModel):
    username: str
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str