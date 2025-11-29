import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)

# переменная для кеширования экземпляра S3Storage
_storage_instance: Optional["S3Storage"] = None


class S3StorageError(Exception):
    """Базовое исключение для операций с S3/MinIO."""


class S3Storage:
    """Обёртка над boto3 для хранения артефактов моделей в S3/MinIO."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        region_name: Optional[str],
        access_key: str,
        secret_key: str,
        bucket_name: str,
        models_prefix: str,
    ) -> None:
        session = boto3.session.Session()
        self.resource = session.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )
        self.bucket = self.resource.Bucket(bucket_name)
        self.models_prefix = models_prefix.strip("/ ")


    def _model_key(self, model_id: str) -> str:
        """Формирует ключ объекта в бакете по ID модели."""
        filename = f"{model_id}.joblib"
        if self.models_prefix:
            return f"{self.models_prefix}/{filename}"
        return filename


    def upload_model(self, model_id: str, local_path: Path) -> None:
        """Загружает локальный файл модели в бакет."""
        key = self._model_key(model_id)
        try:
            self.bucket.upload_file(str(local_path), key)
            logger.info("Модель %s загружена в S3 с ключом %s", model_id, key)
        except (ClientError, BotoCoreError) as e:
            raise S3StorageError(f"Не удалось загрузить модель {model_id} в S3") from e


    def download_model(self, model_id: str, destination: Path) -> bool:
        """Скачивает модель из бакета в указанный путь."""
        key = self._model_key(model_id)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.bucket.download_file(key, str(destination))
            logger.info("Модель %s скачана из S3", model_id)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey"}:
                logger.warning("Модель %s отсутствует в S3", model_id)
                return False
            raise S3StorageError(
                f"Не удалось скачать модель {model_id} из S3"
            ) from e
        except BotoCoreError as e:
            raise S3StorageError(
                f"Не удалось скачать модель {model_id} из S3"
            ) from e


    def delete_model(self, model_id: str) -> None:
        """Удаляет артефакт модели из бакета."""
        key = self._model_key(model_id)
        try:
            self.bucket.Object(key).delete()
            logger.info("Модель %s удалена из S3", model_id)
        except (ClientError, BotoCoreError) as e:
            raise S3StorageError(f"Не удалось удалить модель {model_id} из S3") from e


def _is_s3_configured() -> bool:
    """Проверяет, включена ли работа с S3 и заданы ли все параметры."""
    return all(
        [
            settings.S3_ENABLED,
            settings.S3_ENDPOINT,
            settings.S3_ACCESS_KEY,
            settings.S3_SECRET_KEY,
            settings.S3_BUCKET,
        ]
    )


def _get_storage_instance() -> S3Storage:
    """Возвращает (или создаёт) экземпляр клиента S3."""
    global _storage_instance
    
    if _storage_instance is not None:
        return _storage_instance
    
    if not _is_s3_configured():
        raise S3StorageError("Интеграция с S3/MinIO не настроена")

    _storage_instance = S3Storage(
        endpoint_url=settings.S3_ENDPOINT or "",
        region_name=settings.S3_REGION,
        access_key=settings.S3_ACCESS_KEY or "",
        secret_key=settings.S3_SECRET_KEY or "",
        bucket_name=settings.S3_BUCKET or "",
        models_prefix=settings.S3_MODELS_PREFIX,
    )
    return _storage_instance


def get_storage() -> Optional[S3Storage]:
    """Безопасно возвращает клиент S3, либо None, если интеграция отключена."""
    if not _is_s3_configured():
        return None
    try:
        return _get_storage_instance()
    except S3StorageError as e:
        logger.error("Не удалось инициализировать подключение к S3: %s", e)
        return None


def upload_model_artifact(model_id: str, model_path: Path) -> None:
    """Загружает обученную модель в S3, если интеграция активна."""
    storage = get_storage()
    if not storage:
        return

    try:
        storage.upload_model(model_id, model_path)
    except S3StorageError as e:
        logger.warning("Не удалось загрузить модель %s в S3: %s", model_id, e)


def download_model_artifact(model_id: str, destination: Path) -> bool:
    """Пытается скачать модель из S3, возвращает True при успехе."""
    storage = get_storage()
    if not storage:
        return False

    try:
        return storage.download_model(model_id, destination)
    except S3StorageError as e:
        logger.warning("Не удалось скачать модель %s из S3: %s", model_id, e)
        return False


def delete_model_artifact(model_id: str) -> None:
    """Удаляет модель из S3, если это возможно."""
    storage = get_storage()
    if not storage:
        return

    try:
        storage.delete_model(model_id)
    except S3StorageError as e:
        logger.warning("Не удалось удалить модель %s из S3: %s", model_id, e)