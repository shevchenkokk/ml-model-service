from app.schemas.schemas import TrainModelRequest


def test_train_model_request_valid():
    """Тест валидной схемы TrainModelRequest."""
    data = {
        "model_name": "RandomForest",
        "features": [[1.0, 2.0], [3.0, 4.0]],
        "target": [0, 1],
        "hyperparameters": {"n_estimators": 100}
    }
    request = TrainModelRequest(**data)
    
    assert request.model_name == "RandomForest"
    assert len(request.features) == 2
    assert request.hyperparameters["n_estimators"] == 100
