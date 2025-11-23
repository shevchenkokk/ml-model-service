import logging
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator

from app.core.config import settings

logger = logging.getLogger(__name__)


class MLflowTrackingError(Exception):
    """Базовое исключение для операций с MLflow."""


def _is_mlflow_configured() -> bool:
    """Проверяет, настроен ли MLflow."""
    return settings.MLFLOW_TRACKING_URI is not None


def _setup_mlflow() -> None:
    """Настраивает подключение к MLflow."""
    if not _is_mlflow_configured():
        return

    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow настроен: %s", settings.MLFLOW_TRACKING_URI)
    except Exception as e:
        logger.warning("Не удалось настроить MLflow: %s", e)


def log_training_run(
    model_id: str,
    model_name: str,
    model: BaseEstimator,
    hyperparameters: dict[str, Any],
    metrics: Optional[dict[str, float]] = None,
    model_path: Optional[Path] = None,
) -> None:
    """
    Логирует обучение модели в MLflow.
    """
    if not _is_mlflow_configured():
        logger.debug("MLflow не настроен, пропуск логирования")
        return

    try:
        _setup_mlflow()

        # Используем полный model_id для уникальности (убираем дефисы для читаемости)
        run_name = f"{model_name}_{model_id.replace('-', '')}"
        with mlflow.start_run(run_name=run_name):
            # логируем параметры
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("model_name", model_name)
            for key, value in hyperparameters.items():
                mlflow.log_param(key, str(value))

            # логируем метрики
            if metrics:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

            # логируем модель
            if model_path and model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )

            logger.info("Обучение модели %s залогировано в MLflow", model_id)

    except Exception as e:
        logger.warning("Не удалось залогировать обучение в MLflow: %s", e)


def log_retraining_run(
    model_id: str,
    model_name: str,
    model: BaseEstimator,
    hyperparameters: dict[str, Any],
    metrics: Optional[dict[str, float]] = None,
    model_path: Optional[Path] = None,
) -> None:
    """
    Логирует переобучение модели в MLflow.
    """
    if not _is_mlflow_configured():
        logger.debug("MLflow не настроен, пропуск логирования")
        return

    try:
        _setup_mlflow()

        run_name = f"retrain_{model_name}_{model_id.replace('-', '')}"
        with mlflow.start_run(run_name=run_name):
            # логируем параметры
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("run_type", "retrain")
            for key, value in hyperparameters.items():
                mlflow.log_param(key, str(value))

            # логируем метрики
            if metrics:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

            # логируем модель
            if model_path and model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"{model_name}_retrained",
                )

            logger.info("Переобучение модели %s залогировано в MLflow", model_id)

    except Exception as e:
        logger.warning("Не удалось залогировать переобучение в MLflow: %s", e)