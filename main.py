from fastapi import FastAPI

app = FastAPI(
    title="ML Model Service",
    description="API для обучения и использования ML-моделей",
    version="0.1.0"
)


@app.get("/status")
def get_status():
    """
    Возвращает статус работы сервиса.
    """
    return {"status": "ok"}