import grpc
from concurrent import futures
import logging
import uuid
import joblib
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from generated import (
    ml_model_service_pb2,
    ml_model_service_pb2_grpc
)
from app.core.config import settings
from app.database.database import (
    init_database,
    add_model_to_database,
    get_trained_models_from_database,
    get_model_from_database,
    delete_model_from_database,
    update_model_in_database
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = settings.TRAINED_MODELS_DIR

AVAILABLE_MODELS = {
    "Логистическая регрессия": "logistic_regression",
    "Случайный лес": "random_forest",
    "Градиентный бустинг (LightGBM)": "lightgbm"
}


class MLModelService(ml_model_service_pb2_grpc.MLModelServiceServicer):
    def GetAvailableModels(self, request, context):
        """
        Процедура для получения всех доступных моделей
        """
        logger.info("Запрошен список доступных моделей")
        return ml_model_service_pb2.GetAvailableModelsResponse(models=AVAILABLE_MODELS)


    def TrainModel(self, request, context):
        """
        Процедура для обучения модели
        """
        model_name = request.model_name
        logger.info(f"Получен запрос на обучение модели: {model_name}")

        if model_name not in AVAILABLE_MODELS.values():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Модель '{model_name}' не поддерживается")
            return ml_model_service_pb2.TrainResponse()

        if model_name == "logistic_regression":
            model_cls = LogisticRegression
        elif model_name == "random_forest":
            model_cls = RandomForestClassifier
        else:
            model_cls = LGBMClassifier

        try:
            hyperparameters = MessageToDict(request.hyperparameters)
            int_params = ["n_estimators", "max_depth", "num_leaves", "random_state"]
            for param in int_params:
                if param in hyperparameters:
                    hyperparameters[param] = int(hyperparameters[param])
            features = [list(feature.values) for feature in request.features]
            target = list(request.target)

            # создаём инстанс модели и фитим на переданные данные
            model = model_cls(**hyperparameters)
            model.fit(features, target)

            # генерируем id и сохраняем обученную модель в файл
            model_id = str(uuid.uuid4())
            model_path = TRAINED_MODELS_DIR / f"{model_id}.joblib"
            joblib.dump(model, model_path)

            # сохраняем запись об обученной модели в локальную БД
            add_model_to_database(
                model_id=model_id,
                model_name=model_name,
                hyperparameters=hyperparameters,
                model_path=model_path
            )

            logger.info(f"Обучение модели '{model_name}' завершено. ID модели: {model_id}")

            return ml_model_service_pb2.TrainModelResponse(
                message=f"Модель '{model_name}' успешно обучена. ID: {model_id}",
                trained_model_id=model_id
            )
        except Exception as e:
            logger.error(f"Возникла ошибка при обучении модели: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Возникла ошибка при обучении модели: {e}")
            return ml_model_service_pb2.TrainModelResponse()


    def GetTrainedModels(self, request, context):
        """
        Процедура для получения списка всех обученных моделей
        """
        logger.info("Запрошен список обученных моделей")
        models_from_database = get_trained_models_from_database()

        resp_models = []
        for model_info in models_from_database:
            hyperparams_struct = Struct()
            hyperparams_struct.update(model_info["hyperparameters"])
            resp_models.append(ml_model_service_pb2.TrainedModelInfo(
                id=model_info["id"],
                model_name=model_info["model_name"],
                hyperparameters=hyperparams_struct,
                model_path=model_info["model_path"]
            ))
        return ml_model_service_pb2.GetTrainedModelsResponse(models=resp_models)
    

    def Predict(self, request, context):
        """
        Процедура для получения предсказаний
        """
        model_id = request.model_id

        logger.info(f"Получен запрос на получение предсказаний для модели с ID: '{model_id}'")
        model_info = get_model_from_database(model_id)
        if not model_info:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Модель с ID '{model_id}' не найдена")
            return ml_model_service_pb2.ModelPredictResponse()
        try:
            model_path = model_info["model_path"]
            model = joblib.load(model_path)

            features = [list(feature.values) for feature in request.features]
            preds = model.predict(features)

            return ml_model_service_pb2.ModelPredictResponse(
                model_id=model_id,
                preds=preds.tolist()
            )
        except Exception as e:
            logger.error(f"Ошибка при получении предсказаний для модели с ID '{model_id}': {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ошибка при получении предсказаний: {e}")
            return ml_model_service_pb2.ModelPredictResponse()


    def DeleteModel(self, request, context):
        """
        Процедура для удаления существующей модели
        """
        model_id = request.model_id
        logger.info(f"Получен запрос на удаление модели с ID: '{model_id}'")
        model_info = get_model_from_database(model_id)
        if not model_info:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Модель с ID '{model_id}' не найдена")
            return ml_model_service_pb2.DeleteModelResponse()

        delete_model_from_database(model_id)
        model_path = Path(model_info["model_path"])
        model_path.unlink(missing_ok=True)
        logger.info(f"Файл модели '{model_path}' успешно удален")

        return ml_model_service_pb2.DeleteModelResponse(message=f"Модель с ID '{model_id}' успешно удалена")


    def RetrainModel(self, request, context):
        """
        Процедура для переобучения существующей модели
        """
        model_id = request.model_id
        logger.info(f"Получен запрос на переобучение модели с ID: '{model_id}'")

        model_info = get_model_from_database(model_id)
        if not model_info:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Модель с ID '{model_id}' не найдена")
            return ml_model_service_pb2.ModelRetrainResponse()

        model_name = model_info["model_name"]
        if model_name == "logistic_regression":
            model_cls = LogisticRegression
        elif model_name == "random_forest":
            model_cls = RandomForestClassifier
        else:
            model_cls = LGBMClassifier

        try:
            hyperparameters = MessageToDict(request.hyperparameters)
            int_params = ["n_estimators", "max_depth", "num_leaves", "random_state"]
            for param in int_params:
                if param in hyperparameters:
                    hyperparameters[param] = int(hyperparameters[param])
            features = [list(feature.values) for feature in request.features]
            target = list(request.target)

            # создаём инстанс модели и фитим на переданные данные
            model = model_cls(**hyperparameters)
            model.fit(features, target)

            # перезаписываем файл модели
            model_path = TRAINED_MODELS_DIR / f"{model_id}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Файл модели {model_path} успешно перезаписан")

            # обновляем инфу по модели в БД
            update_model_in_database(model_id, hyperparameters)

            logger.info(f"Переобучение модели '{model_name}' завершено. ID модели: {model_id}")

            return ml_model_service_pb2.RetrainModelResponse(
                message=f"Модель с ID '{model_id}' успешно переобучена"
            )
        except Exception as e:
            logger.error(f"Ошибка при переобучении модели с ID '{model_id}': {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Возникла ошибка при переобучении модели: {e}")
            return ml_model_service_pb2.RetrainResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_model_service_pb2_grpc.add_MLModelServiceServicer_to_server(MLModelService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("gRPC сервер запущен на порту 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    init_database()
    logger.info("База данных успешно инициализирована")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Создана папка '{TRAINED_MODELS_DIR}' для хранения обученных моделей")
    serve()