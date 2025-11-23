import logging
import uuid
from pathlib import Path
from typing import Any, Sequence

import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.core.config import settings
from app.database.database import (
    add_model_to_database,
    delete_model_from_database,
    get_model_from_database,
    get_trained_models_from_database,
    update_model_in_database,
)
from app.storage.s3 import (
    delete_model_artifact,
    download_model_artifact,
    upload_model_artifact,
)
from app.storage.dvc import save_dataset, version_dataset

logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = settings.TRAINED_MODELS_DIR

AVAILABLE_MODELS = {
    "Логистическая регрессия": "logistic_regression",
    "Случайный лес": "random_forest",
    "Градиентный бустинг (LightGBM)": "lightgbm",
}


class ModelServiceError(Exception):
    """Базовое исключение сервиса работы с моделями."""


class ModelNotSupportedError(ModelServiceError):
    """Модель с таким именем не поддерживается."""


class TrainedModelNotFoundError(ModelServiceError):
    """Обученная модель не найдена в БД."""


class ModelFileMissingError(ModelServiceError):
    """Файл обученной модели отсутствует на диске."""


def _get_model_class(model_name: str):
    """Возвращает класс модели по её имени."""
    if model_name == "logistic_regression":
        return LogisticRegression
    if model_name == "random_forest":
        return RandomForestClassifier
    if model_name == "lightgbm":
        return LGBMClassifier
    raise ModelNotSupportedError(f"Модель '{model_name}' не поддерживается")


def train_model(
    model_name: str,
    hyperparameters: dict[str, Any],
    features: Sequence[Sequence[float]],
    target: Sequence[int],
) -> str:
    """Обучает модель и сохраняет её."""
    model_id = str(uuid.uuid4())
    
    # cохраняем датасет и версионируем через DVC
    try:
        dataset_path = save_dataset(model_id, list(features), list(target), dataset_type="train")
        version_dataset(dataset_path)
    except Exception as e:
        logger.warning("Не удалось сохранить датасет через DVC: %s", e)
    
    model_cls = _get_model_class(model_name)

    try:
        model = model_cls(**hyperparameters)
        model.fit(features, target)
    except Exception as e:
        logger.exception("Ошибка при обучении модели %s", model_name)
        raise ModelServiceError(f"Ошибка при обучении модели: {e}") from e

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TRAINED_MODELS_DIR / f"{model_id}.joblib"
    joblib.dump(model, model_path)
    upload_model_artifact(model_id, model_path)

    add_model_to_database(
        model_id=model_id,
        model_name=model_name,
        hyperparameters=hyperparameters,
        model_path=model_path,
    )
    logger.info("Модель '%s' обучена. ID: %s", model_name, model_id)
    return model_id


def list_trained_models():
    """Возвращает список обученных моделей."""
    return get_trained_models_from_database()


def _get_model_info_or_raise(model_id: str) -> dict[str, Any]:
    "Получает информацию о модели по её ID или выбрасывает исключение."
    model_info = get_model_from_database(model_id)
    if not model_info:
        raise TrainedModelNotFoundError(f"Модель с ID '{model_id}' не найдена")
    return model_info


def predict_model(model_id: str, features: Sequence[Sequence[float]]) -> list[int]:
    """Вычисляет предсказания с использованием обученной модели."""
    model_info = _get_model_info_or_raise(model_id)
    model_path = Path(model_info["model_path"])

    # если файл отсутствует локально, пытаемся скачать из S3
    if not model_path.exists():
        fetched = download_model_artifact(model_id, model_path)
        if not fetched:
            logger.error("Файл для модели '%s' не найден локально и в S3: %s", model_id, model_path)
            raise ModelFileMissingError("Файл модели не найден на сервере")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.exception("Ошибка при загрузке модели %s из файла %s", model_id, model_path)
        raise ModelFileMissingError(f"Не удалось загрузить модель из файла: {e}") from e

    try:
        preds = model.predict(features)
    except Exception as e:
        logger.exception("Ошибка при получении предсказаний модели %s", model_id)
        raise ModelServiceError(f"Ошибка при получении предсказаний: {e}") from e

    return preds.tolist()


def delete_trained_model(model_id: str) -> None:
    """Удаляет обученную модель из файловой системы и БД."""
    model_info = _get_model_info_or_raise(model_id)

    deleted_rows = delete_model_from_database(model_id)
    if deleted_rows == 0:
        raise TrainedModelNotFoundError(f"Модель с ID '{model_id}' не найдена")

    model_path = Path(model_info["model_path"])
    model_path.unlink(missing_ok=True)
    delete_model_artifact(model_id)
    logger.info("Модель %s успешно удалена", model_id)


def retrain_model(
    model_id: str,
    hyperparameters: dict[str, Any],
    features: Sequence[Sequence[float]],
    target: Sequence[int],
) -> None:
    model_info = _get_model_info_or_raise(model_id)
    
    # cохраняем датасет для переобучения и версионируем через DVC
    try:
        dataset_path = save_dataset(model_id, list(features), list(target), dataset_type="retrain")
        version_dataset(dataset_path)
    except Exception as e:
        logger.warning("Не удалось сохранить датасет переобучения через DVC: %s", e)
    
    model_cls = _get_model_class(model_info["model_name"])

    try:
        new_model = model_cls(**hyperparameters)
        new_model.fit(features, target)
    except Exception as e:
        logger.exception("Ошибка при переобучении модели %s", model_id)
        raise ModelServiceError(f"Ошибка при переобучении модели: {e}") from e

    model_path = Path(model_info["model_path"])
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_model, model_path)
    upload_model_artifact(model_id, model_path)
    update_model_in_database(model_id, hyperparameters)
    logger.info("Модель %s успешно переобучена", model_id)