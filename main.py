from fastapi import FastAPI
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


@app.on_event("startup")
async def startup_event():
    """
    Логгирует сообщение при старте сервиса.
    """
    logger.info("Сервис запущен")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Логгирует сообщение при остановке сервиса.
    """
    logger.info("Сервис остановлен")


@app.get("/status")
def get_status():
    """
    Возвращает статус работы сервиса.
    """
    logger.info("Запрос на эндпоинт /status")
    return {"status": "ok"}