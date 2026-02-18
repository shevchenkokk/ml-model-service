import logging

from fastapi import FastAPI

from app.api.api import router as api_router
from app.core.config import settings
from app.database.database import create_users_table, init_database
from app.storage.dvc import setup_dvc_remote

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Service",
    description="API для обучения и использования ML-моделей",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """
    Логирует сообщение при старте сервиса.
    """
    logger.info("Сервис запущен")
    init_database()
    create_users_table()
    settings.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # настраиваем DVC remote для версионирования датасетов
    setup_dvc_remote()

    logger.info("База данных и директория с моделями готовы к работе")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Логирует остановку сервиса.
    """
    logger.info("Сервис остановлен")


app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """
    Health check endpoint для мониторинга состояния сервиса.
    """
    return {
        "status": "healthy",
        "service": "ml-model-service",
        "version": "0.1.0"
    }