import logging
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# путь к директории с данными для DVC
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


class DVCError(Exception):
    """Базовое исключение для операций с DVC."""


def _create_data_dirs() -> None:
    """Создаёт необходимые директории для данных, если их нет."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _run_dvc_command(cmd: list[str], cwd: Optional[Path] = None) -> tuple[str, str, int]:
    """
    Выполняет команду DVC и возвращает stdout, stderr и код возврата.
    """
    try:
        result = subprocess.run(
            ["dvc"] + cmd,
            capture_output=True,
            text=True,
            cwd=cwd or Path.cwd(),
            check=False,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        raise DVCError("DVC не установлен. Установите его: pip install dvc[s3]")


def save_dataset(
    dataset_id: str,
    features: list[list[float]],
    target: list[int],
    dataset_type: str = "train",
) -> Path:
    """
    Сохраняет датасет в файл и возвращает путь к нему.
    """
    _create_data_dirs()

    output_dir = PROCESSED_DATA_DIR / dataset_type
    output_dir.mkdir(parents=True, exist_ok=True)

    df_features = pd.DataFrame(features)
    df_target = pd.DataFrame({"target": target})
    df = pd.concat([df_features, df_target], axis=1)

    filepath = output_dir / f"{dataset_id}.csv"
    df.to_csv(filepath, index=False)

    logger.info("Датасет сохранён: %s", filepath)
    return filepath


def version_dataset(dataset_path: Path) -> None:
    """
    Версионирует датасет через DVC и отправляет в MinIO.
    """
    if not settings.S3_ENABLED:
        logger.debug("S3 отключен, пропуск версионирования через DVC")
        return
    
    try:
        # добавляем файл в DVC
        stdout, stderr, return_code = _run_dvc_command(["add", str(dataset_path)])
        
        if return_code != 0:
            logger.warning(
                "Не удалось добавить датасет в DVC: %s. stderr: %s",
                dataset_path,
                stderr,
            )
            return
        
        logger.info("Датасет добавлен в DVC: %s", dataset_path)
        
        # отправляем данные в MinIO через DVC remote
        stdout, stderr, return_code = _run_dvc_command(["push"])
        if return_code == 0:
            logger.info("Датасет отправлен в DVC remote (MinIO)")
        else:
            logger.warning(
                "Не удалось отправить датасет в DVC remote: %s. "
                "Возможно, remote не настроен или MinIO недоступен.",
                stderr,
            )     
    except DVCError as e:
        logger.warning("Ошибка при версионировании датасета через DVC: %s", e)


def setup_dvc_remote() -> None:
    """
    Настраивает DVC remote на MinIO для хранения версионированных данных.
    """
    if not settings.S3_ENABLED:
        logger.debug("S3 отключен, пропускаем настройку DVC remote")
        return
    
    if not all([settings.S3_ENDPOINT, settings.S3_BUCKET, settings.S3_ACCESS_KEY, settings.S3_SECRET_KEY]):
        logger.warning("Не все параметры S3 настроены, пропускаем настройку DVC remote")
        return
    
    try:
        # проверка, есть ли уже remote
        stdout, stderr, return_code = _run_dvc_command(["remote", "list"])
        if return_code == 0 and ("minio" in stdout.lower() or "s3" in stdout.lower()):
            logger.info("DVC remote уже настроен")
            return

        endpoint = settings.S3_ENDPOINT
        bucket = settings.S3_BUCKET
        
        # формируем URL для MinIO
        remote_url = f"s3://{bucket}/dvc"
        
        # добавляем remote
        stdout, stderr, return_code = _run_dvc_command(["remote", "add", "-d", "minio", remote_url])
        if return_code != 0:
            logger.warning("Не удалось добавить DVC remote: %s", stderr)
            return
        
        # настройка параметров подключения
        _run_dvc_command(["remote", "modify", "minio", "endpointurl", endpoint])
        _run_dvc_command(["remote", "modify", "minio", "access_key_id", settings.S3_ACCESS_KEY])
        _run_dvc_command(["remote", "modify", "minio", "secret_access_key", settings.S3_SECRET_KEY])
        
        logger.info("DVC remote 'minio' настроен на %s", endpoint)
        
    except DVCError as e:
        logger.warning("Не удалось настроить DVC remote: %s", e)