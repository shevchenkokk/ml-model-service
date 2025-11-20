import logging
import uuid
from pathlib import Path

import joblib
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.auth.auth import (
    create_access_token,
    get_current_user,
    get_user,
    verify_password,
)
from app.core.config import settings
from app.database.database import (
    add_model_to_database,
    create_user_in_database,
    delete_model_from_database,
    get_model_from_database,
    get_trained_models_from_database,
    update_model_in_database,
)
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

logger = logging.getLogger(__name__)
router = APIRouter()

AVAILABLE_MODELS = {
    "Логистическая регрессия": "logistic_regression",
    "Случайный лес": "random_forest",
    "Градиентный бустинг (LightGBM)": "lightgbm",
}

TRAINED_MODELS_DIR = settings.TRAINED_MODELS_DIR


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

    # проверка, что модель в списке доступных
    if req.model_name not in AVAILABLE_MODELS.values():
        raise HTTPException(status_code=400, detail=f"Модель '{req.model_name}' не поддерживается")

    if req.model_name == "logistic_regression":
        model_cls = LogisticRegression
    elif req.model_name == "random_forest":
        model_cls = RandomForestClassifier
    else:
        model_cls = LGBMClassifier

    try:
        # создаём инстанс модели и фитим на переданные данные
        model = model_cls(**req.hyperparameters)
        model.fit(req.features, req.target)

        # генерируем id и сохраняем обученную модель в файл
        model_id = str(uuid.uuid4())
        model_path = TRAINED_MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(model, model_path)

        # сохраняем запись об обученной модели в локальную БД
        add_model_to_database(
            model_id=model_id,
            model_name=req.model_name,
            hyperparameters=req.hyperparameters,
            model_path=model_path
        )

        logger.info(f"Обучение модели '{req.model_name}' завершено. ID модели: {model_id}")

        return TrainModelResponse(
            message=f"Модель '{req.model_name}' успешно обучена. ID: {model_id}",
            trained_model_id=model_id
        )

    except Exception as e:
        # если что-то не так при обучении (некорректное имя модели, гиперпараметры и т.д.) -> выбрасываем ошибку
        logger.error(f"Возникла ошибка при обучении модели: {e}")
        raise HTTPException(
            status_code=500, detail=f"Возникла ошибка при обучении модели: {e}"
        )


@router.get("/trained-models", response_model=list[TrainedModelInfo])
async def get_trained_models(current_user: User = Depends(get_current_user)):
    """
    Возвращает список всех обученных моделей.
    """
    logger.info("Запрошен список обученных моделей")
    trained_models = get_trained_models_from_database()
    return trained_models


@router.post("/predict/{model_id}", response_model=ModelPredictResponse)
async def predict(
    model_id: str, req: ModelPredictRequest, current_user: User = Depends(get_current_user)
):
    """
    Возвращает предсказания модели с ID `model_id` на переданных данных
    """
    logger.info(f"Получен запрос на получение предсказаний для модели с ID: '{model_id}'")
    model_info = get_model_from_database(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена")
    try:
        model_path = model_info["model_path"]
        model = joblib.load(model_path)

        preds = model.predict(req.features)

        return ModelPredictResponse(
            model_id=model_id,
            preds=preds.tolist()
        )

    except FileNotFoundError:
        logger.error(f"Файл для модели '{model_id}' не найден по пути {model_path}")
        raise HTTPException(status_code=404, detail="Файл модели не найден на сервере")
    except Exception as e:
        logger.error(f"Ошибка при получении предсказаний для модели с ID '{model_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении предсказаний: {e}")


@router.delete("/trained-models/{model_id}", status_code=200)
async def delete_trained_model(
    model_id: str, current_user: User = Depends(get_current_user)
):
    """
    Удаляет обученную модель: стирает файл и запись с БД.
    """
    logger.info(f"Получен запрос на удаление модели с ID: '{model_id}'")
    model_info = get_model_from_database(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена")
    
    # удаляем запись о модели из БД
    deleted_rows_num = delete_model_from_database(model_id)
    if deleted_rows_num == 0:
        raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена")

    # удаляем файл модели с диска
    try:
        model_path = Path(model_info["model_path"])
        model_path.unlink(missing_ok=True)
        logger.info(f"Файл модели '{model_path}' успешно удален")
    except Exception as e:
        logger.error(f"Не удалось удалить файл модели {model_path}: {e}")
    
    return {"message": f"Модель с ID '{model_id}' успешно удалена"}


@router.put("/retrain/{model_id}")
async def retrain_model(
    model_id: str, req: RetrainModelRequest, current_user: User = Depends(get_current_user)
):
    """
    Переобучает уже существующую модель на новых данных.
    """
    logger.info(f"Получен запрос на переобучение модели с ID: '{model_id}'")

    model_info = get_model_from_database(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена")

    model_name = model_info["model_name"]
    if model_name == "logistic_regression":
        model_cls = LogisticRegression
    elif model_name == "random_forest":
        model_cls = RandomForestClassifier
    else:
        model_cls = LGBMClassifier

    try:
        new_model = model_cls(**req.hyperparameters)
        new_model.fit(req.features, req.target)

        # перезаписываем старый файл модели новым
        model_path = Path(model_info["model_path"])
        joblib.dump(new_model, model_path)
        logger.info(f"Файл модели {model_path} успешно перезаписан")

        # обновляем инфу по модели в БД
        update_model_in_database(model_id, req.hyperparameters)

        return {"message": f"Модель с ID '{model_id}' успешно переобучена"}
    except Exception as e:
        logger.error(f"Ошибка при переобучении модели с ID '{model_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при переобучении модели: {e}")


@router.post("/register", status_code=201)
async def register_user(user: UserCreate):
    """
    Регистрирует нового пользователя в системе
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