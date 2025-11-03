from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Any
from pathlib import Path
import uuid
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from database import (
    init_database,
    add_model_to_database,
    get_trained_models_from_database,
    get_model_from_database,
    TRAINED_MODELS_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Service",
    description="API для обучения и использования ML-моделей",
    version="0.1.0"
)

AVAILABLE_MODELS = {
    "Логистическая регрессия": "logistic_regression",
    "Случайный лес": "random_forest",
    "Градиентный бустинг (LightGBM)": "lightgbm"
}


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


@app.on_event("startup")
async def startup_event():
    """
    Логирует сообщение при старте сервиса.
    """
    logger.info("Сервис запущен")
    # инициализация БД
    init_database()
    logger.info("База данных успешно инициализирована")
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    logger.info(f"Создана папка '{TRAINED_MODELS_DIR}' для хранения обученных моделей")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Логирует сообщение при остановке сервиса.
    """
    logger.info("Сервис остановлен")


@app.get("/status")
def get_status():
    """
    Возвращает статус работы сервиса.
    """
    logger.info("Запрос на эндпоинт /status")
    return {"status": "ok"}


@app.get("/models", response_model=AvailableModelsResponse)
def get_available_models():
    """
    Возвращает список доступных для обучения классов моделей.
    """
    logger.info("Запрошен список доступных моделей")
    return {"available_models": AVAILABLE_MODELS}


@app.post("/train", response_model=TrainModelResponse)
def train_model(req: TrainModelRequest):
    """
    Обучает ML-модель с переданными гиперпараметрами.
    """
    logger.info(f"Получен запрос на обучение модели: {req.model_name}")

    # проверка, что модель в списке доступных
    if req.model_name not in AVAILABLE_MODELS.values():
        raise HTTPException(status_code=400, detail=f"Модель '{req.model_name}' не поддерживается")

    if req.model_name == "logistic_regression":
        model_cls = LogisticRegression
    elif req.model_name == "random_forest":
        model_cls = RandomForestClassifier
    else:
        model_cls = LGBMClassifier

    try:
        # создаём инстанс модели и фитим на переданные данные
        model = model_cls(**req.hyperparameters)
        model.fit(req.features, req.target)

        # генерируем id и сохраняем обученную модель в файл
        model_id = str(uuid.uuid4())
        model_path = TRAINED_MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(model, model_path)

        # сохраняем запись об обученной модели в локальную БД
        add_model_to_database(
            model_id=model_id,
            model_name=req.model_name,
            hyperparameters=req.hyperparameters,
            model_path=model_path
        )

        logger.info(f"Обучение модели '{req.model_name}' завершено. ID модели: {model_id}")

        return TrainModelResponse(
            message=f"Модель '{req.model_name}' успешно обучена",
            trained_model_id=model_id
        )

    except Exception as e:
        # если что-то не так при обучении (некорректное имя модели, гиперпараметры и т.д.) -> выбрасываем ошибку
        logger.error(f"Возникла ошибка при обучении модели: {e}")
        raise HTTPException(status_code=500, detail=f"Возникла ошибка при обучении модели: {e}")


@app.get("/trained-models", response_model=list[TrainedModelInfo])
def get_trained_models():
    """
    Возвращает список всех обученных моделей.
    """
    logger.info("Запрошен список обученных моделей")
    trained_models = get_trained_models_from_database()
    return trained_models


@app.post("/predict/{model_id}", response_model=ModelPredictResponse)
def predict(model_id: str, req: ModelPredictRequest):
    """
    Возвращает предсказания модели с ID `model_id` на переданных данных
    """
    logger.info(f"Получен запрос на получение предсказаний для модели с ID: '{model_id}'")
    model_info = get_model_from_database(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена")
    try:
        model_path = model_info["model_path"]
        model = joblib.load(model_path)

        preds = model.predict(req.features)

        return ModelPredictResponse(
            model_id=model_id,
            preds=preds.tolist()
        )

    except FileNotFoundError:
        logger.error(f"Файл для модели '{model_id}' не найден по пути {model_path}")
        raise HTTPException(status_code=404, detail="Файл модели не найден на сервере")
    except Exception as e:
        logger.error(f"Ошибка при получении предсказаний для модели с ID '{model_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении предсказаний: {e}")