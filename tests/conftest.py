import os
import pytest
from unittest.mock import MagicMock, patch

os.environ["SECRET_KEY"] = "test_secret_key"

from app.storage.s3 import S3Storage


@pytest.fixture
def mock_s3_resource():
    """Мок ресурса сессии boto3."""
    with patch("boto3.session.Session") as mock_session:
        mock_resource = MagicMock()
        mock_session.return_value.resource.return_value = mock_resource
        yield mock_resource


@pytest.fixture
def s3_storage(mock_s3_resource):
    """Фикстура для экземпляра S3Storage с замоканным ресурсом boto3."""
    return S3Storage(
        endpoint_url="http://localhost:9000",
        region_name="us-east-1",
        access_key="minio",
        secret_key="minio123",
        bucket_name="ml-models",
        models_prefix="models"
    )
