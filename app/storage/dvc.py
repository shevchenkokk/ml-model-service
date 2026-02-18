import logging
import subprocess
import yaml
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


def _is_dvc_initialized() -> bool:
    """Проверяет, инициализирован ли DVC в проекте."""
    dvc_dir = Path(".dvc")
    return dvc_dir.exists() and (dvc_dir / "config").exists()


def init_dvc() -> None:
    """
    Инициализирует DVC в проекте, если он ещё не инициализирован.
    """
    if _is_dvc_initialized():
        logger.debug("DVC уже инициализирован")
        return
    
    if not settings.S3_ENABLED:
        logger.debug("S3 отключен, пропускаем инициализацию DVC")
        return
    
    try:
        # инициализируем DVC без git (--no-scm)
        stdout, stderr, return_code = _run_dvc_command(["init", "--no-scm"])
        
        if return_code == 0:
            logger.info("DVC успешно инициализирован")
        else:
            logger.warning(
                "Не удалось инициализировать DVC: %s. stderr: %s",
                stdout,
                stderr,
            )
    except DVCError as e:
        logger.warning("Ошибка при инициализации DVC: %s", e)


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


def version_dataset(dataset_path: Path) -> Optional[str]:
    """
    Версионирует датасет через DVC и отправляет в MinIO.
    """
    if not settings.S3_ENABLED:
        logger.debug("S3 отключен, пропуск версионирования через DVC")
        return None

    # гарантируем настройку remote перед использованием
    setup_dvc_remote()
    
    # проверяем, инициализирован ли DVC
    if not _is_dvc_initialized():
        logger.warning("DVC не инициализирован, пропуск версионирования датасета")
        return None

    data_hash = None

    try:
        # добавляем файл в DVC
        stdout, stderr, return_code = _run_dvc_command(["add", str(dataset_path)])
        
        if return_code != 0:
            logger.warning(
                "Не удалось добавить датасет в DVC: %s. stderr: %s",
                dataset_path,
                stderr,
            )
            return None
        
        logger.info("Датасет добавлен в DVC: %s", dataset_path)

        # читаем .dvc файл, чтобы узнать хэш версии
        dvc_file_path = dataset_path.parent / (dataset_path.name + ".dvc")
        
        if dvc_file_path.exists():
            try:
                with open(dvc_file_path, "r") as f:
                    dvc_meta = yaml.safe_load(f)
                    if "outs" in dvc_meta and len(dvc_meta["outs"]) > 0:
                        data_hash = dvc_meta["outs"][0].get("md5")
                        logger.info(f"Получен хеш версии данных: {data_hash}")
            except Exception as e:
                logger.error(f"Ошибка при чтении .dvc файла: {e}")

        # отправляем данные в MinIO через DVC remote
        stdout, stderr, return_code = _run_dvc_command(["push", str(dvc_file_path)])
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
    
    return data_hash


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
    
    # инициализируем DVC, если ещё не инициализирован
    init_dvc()
    
    # если DVC не инициализирован, не можем настроить remote
    if not _is_dvc_initialized():
        logger.warning("DVC не инициализирован, пропускаем настройку remote")
        return
    
    try:
        endpoint = settings.S3_ENDPOINT
        bucket = settings.S3_BUCKET
        
        # формируем URL для MinIO
        remote_url = f"s3://{bucket}/dvc"

        _run_dvc_command(["remote", "add", "-d", "-f", "minio", remote_url])
        
        # настройка параметров подключения
        _run_dvc_command(["remote", "modify", "minio", "endpointurl", endpoint])
        _run_dvc_command(["remote", "modify", "minio", "access_key_id", settings.S3_ACCESS_KEY])
        _run_dvc_command(["remote", "modify", "minio", "secret_access_key", settings.S3_SECRET_KEY])
        _run_dvc_command(["remote", "modify", "minio", "ssl_verify", "false"])
        
        logger.info("DVC remote 'minio' настроен на %s", endpoint)
        
    except DVCError as e:
        logger.warning("Не удалось настроить DVC remote: %s", e)