from fastapi import FastAPI
from pydantic import BaseModel
import logging

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

available_models = {
    "Логистическая регрессия": "logistic_regression",
    "Случайный лес": "random_forest",
    "Градиентный бустинг (LightGBM)": "lightgbm"
}

class AvailableModelsResponse(BaseModel):
    available_models: dict[str, str]


@app.on_event("startup")
async def startup_event():
    """
    Логирует сообщение при старте сервиса.
    """
    logger.info("Сервис запущен")


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
    return {"available_models": available_models}