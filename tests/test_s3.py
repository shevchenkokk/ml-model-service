import uuid
from pathlib import Path
from unittest.mock import patch


def test_upload_model(s3_storage):
    """Тест загрузки модели в S3 с использованием замоканного хранилища."""
    model_id = str(uuid.uuid4())
    local_path = Path("/tmp/test_model.joblib")

    s3_storage.upload_model(model_id, local_path)

    expected_key = f"models/{model_id}.joblib"

    s3_storage.bucket.upload_file.assert_called_once_with(str(local_path), expected_key)


def test_download_model(s3_storage):
    """Тест скачивания модели из S3."""
    model_id = str(uuid.uuid4())
    destination = Path("/tmp/downloaded_model.joblib")
    
    with patch("pathlib.Path.mkdir"):
        result = s3_storage.download_model(model_id, destination)

        assert result is True
        expected_key = f"models/{model_id}.joblib"
        s3_storage.bucket.download_file.assert_called_once_with(expected_key, str(destination))
