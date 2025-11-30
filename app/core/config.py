from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    DB_FILE: Path = Path("data/ml_model_service.db")
    TRAINED_MODELS_DIR: Path = Path("trained_models")

    # настройки аутентификации
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    # настройки S3/MinIO
    S3_ENABLED: bool = False
    S3_ENDPOINT: str | None = None
    S3_REGION: str | None = None
    S3_ACCESS_KEY: str | None = None
    S3_SECRET_KEY: str | None = None
    S3_BUCKET: str | None = None
    S3_MODELS_PREFIX: str = "models/"

    # настройки MLflow
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT_NAME: str = "ml-model-service"

    class Config:
        env_file = ".env"


settings = Settings()