from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    DB_FILE: Path = Path("ml_model_service.db")
    TRAINED_MODELS_DIR: Path = Path("trained_models")
    
    # настройки аутентификации
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    class Config:
        env_file = ".env"


settings = Settings()