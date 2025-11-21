import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.auth import (
    create_access_token,
    get_current_user,
    get_user,
    verify_password,
)
from app.database.database import create_user_in_database
from app.schemas.schemas import (
    AvailableModelsResponse,
    ModelPredictRequest,
    ModelPredictResponse,
    RetrainModelRequest,
    Token,
    TrainModelRequest,
    TrainModelResponse,
    TrainedModelInfo,
    User,
    UserCreate,
)
from app.services import models as model_service
from app.services.models import (
    AVAILABLE_MODELS,
    ModelFileMissingError,
    ModelNotSupportedError,
    ModelServiceError,
    TrainedModelNotFoundError,
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status")
async def get_status():
    """
    Возвращает статус работы сервиса.
    """
    logger.info("Запрос на эндпоинт /status")
    return {"status": "ok"}


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """
    Возвращает список доступных для обучения классов моделей.
    """
    logger.info("Запрошен список доступных моделей")
    return {"available_models": AVAILABLE_MODELS}


@router.post("/train", response_model=TrainModelResponse)
async def train_model(
    req: TrainModelRequest, current_user: User = Depends(get_current_user)
):
    """
    Обучает ML-модель с переданными гиперпараметрами.
    """
    logger.info(f"Получен запрос на обучение модели: {req.model_name}")

    try:
        model_id = model_service.train_model(
            model_name=req.model_name,
            hyperparameters=req.hyperparameters,
            features=req.features,
            target=req.target,
        )
        logger.info(
            "Обучение модели '%s' завершено. ID модели: %s", req.model_name, model_id
        )

        return TrainModelResponse(
            message=f"Модель '{req.model_name}' успешно обучена. ID: {model_id}",
            trained_model_id=model_id
        )
    except ModelNotSupportedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ModelServiceError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/trained-models", response_model=list[TrainedModelInfo])
async def get_trained_models(current_user: User = Depends(get_current_user)):
    """
    Возвращает список всех обученных моделей.
    """
    logger.info("Запрошен список обученных моделей")
    trained_models = model_service.list_trained_models()
    return trained_models


@router.post("/predict/{model_id}", response_model=ModelPredictResponse)
async def predict(
    model_id: str, req: ModelPredictRequest, current_user: User = Depends(get_current_user)
):
    """
    Возвращает предсказания модели с ID `model_id` на переданных данных.
    """
    logger.info(f"Получен запрос на получение предсказаний для модели с ID: '{model_id}'")
    try:
        preds = model_service.predict_model(model_id=model_id, features=req.features)

        return ModelPredictResponse(
            model_id=model_id,
            preds=preds
        )
    except TrainedModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelFileMissingError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelServiceError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/trained-models/{model_id}", status_code=200)
async def delete_trained_model(
    model_id: str, current_user: User = Depends(get_current_user)
):
    """
    Удаляет обученную модель: стирает файл и запись с БД.
    """
    logger.info(f"Получен запрос на удаление модели с ID: '{model_id}'")
    try:
        model_service.delete_trained_model(model_id)
    except TrainedModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelServiceError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"message": f"Модель с ID '{model_id}' успешно удалена"}


@router.put("/retrain/{model_id}")
async def retrain_model(
    model_id: str, req: RetrainModelRequest, current_user: User = Depends(get_current_user)
):
    """
    Переобучает уже существующую модель на новых данных.
    """
    logger.info(f"Получен запрос на переобучение модели с ID: '{model_id}'")

    try:
        model_service.retrain_model(
            model_id=model_id,
            hyperparameters=req.hyperparameters,
            features=req.features,
            target=req.target,
        )
        return {"message": f"Модель с ID '{model_id}' успешно переобучена"}
    except TrainedModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelServiceError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/register", status_code=201)
async def register_user(user: UserCreate):
    """
    Регистрирует нового пользователя в системе.
    """
    try:
        create_user_in_database(user.username, user.password)
        return {"message": f"Пользователь '{user.username}' успешно зарегистрирован"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка на сервере: {e}")


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Проверяет корректность учетных данных пользователя и возвращает JWT-токен.
    """
    # идентификация пользователя
    user = get_user(form_data.username)
    # проверка, что пароль соответствует тому, что хранится в БД
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Некорректное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"}
        )
    # если всё ок – генерируем токен
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}