import grpc
from concurrent import futures
import logging

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from generated import (
    ml_model_service_pb2,
    ml_model_service_pb2_grpc
)
from app.core.config import settings
from app.database.database import init_database
from app.services import models as model_service
from app.services.models import (
    AVAILABLE_MODELS,
    ModelFileMissingError,
    ModelNotSupportedError,
    ModelServiceError,
    TrainedModelNotFoundError,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = settings.TRAINED_MODELS_DIR


class MLModelService(ml_model_service_pb2_grpc.MLModelServiceServicer):
    def GetAvailableModels(self, request, context):
        """
        Процедура для получения всех доступных моделей.
        """
        logger.info("Запрошен список доступных моделей")
        return ml_model_service_pb2.GetAvailableModelsResponse(models=AVAILABLE_MODELS)


    def TrainModel(self, request, context):
        """
        Процедура для обучения модели.
        """
        model_name = request.model_name
        logger.info(f"Получен запрос на обучение модели: {model_name}")

        try:
            hyperparameters = MessageToDict(request.hyperparameters)
            int_params = ["n_estimators", "max_depth", "num_leaves", "random_state"]
            for param in int_params:
                if param in hyperparameters:
                    hyperparameters[param] = int(hyperparameters[param])
            features = [list(feature.values) for feature in request.features]
            target = list(request.target)

            trained_model_id = model_service.train_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                features=features,
                target=target,
            )
            logger.info(f"Обучение модели '{model_name}' завершено. ID модели: {trained_model_id}")

            return ml_model_service_pb2.TrainModelResponse(
                message=f"Модель '{model_name}' успешно обучена. ID: {trained_model_id}",
                trained_model_id=trained_model_id
            )
        except ModelNotSupportedError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_model_service_pb2.TrainModelResponse()
        except ModelServiceError as e:
            logger.error(f"Возникла ошибка при обучении модели: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_model_service_pb2.TrainModelResponse()


    def GetTrainedModels(self, request, context):
        """
        Процедура для получения списка всех обученных моделей.
        """
        logger.info("Запрошен список обученных моделей")
        models_from_database = model_service.list_trained_models()

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
        Процедура для получения предсказаний.
        """
        model_id = request.model_id

        logger.info(f"Получен запрос на получение предсказаний для модели с ID: '{model_id}'")
        try:
            features = [list(feature.values) for feature in request.features]
            preds = model_service.predict_model(model_id=model_id, features=features)

            return ml_model_service_pb2.ModelPredictResponse(
                model_id=model_id,
                preds=preds
            )
        except TrainedModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_model_service_pb2.ModelPredictResponse()
        except ModelFileMissingError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_model_service_pb2.ModelPredictResponse()
        except ModelServiceError as e:
            logger.error(f"Ошибка при получении предсказаний для модели с ID '{model_id}': {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_model_service_pb2.ModelPredictResponse()


    def DeleteModel(self, request, context):
        """
        Процедура для удаления существующей модели.
        """
        model_id = request.model_id
        logger.info(f"Получен запрос на удаление модели с ID: '{model_id}'")
        try:
            model_service.delete_trained_model(model_id)
            return ml_model_service_pb2.DeleteModelResponse(message=f"Модель с ID '{model_id}' успешно удалена")
        except TrainedModelNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return ml_model_service_pb2.DeleteModelResponse()
        except ModelServiceError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return ml_model_service_pb2.DeleteModelResponse()


    def RetrainModel(self, request, context):
        """
        Процедура для переобучения существующей модели.
        """
        model_id = request.model_id
        logger.info(f"Получен запрос на переобучение модели с ID: '{model_id}'")

        try:
            hyperparameters = MessageToDict(request.hyperparameters)
            int_params = ["n_estimators", "max_depth", "num_leaves", "random_state"]
            for param in int_params:
                if param in hyperparameters:
                    hyperparameters[param] = int(hyperparameters[param])
            features = [list(feature.values) for feature in request.features]
            target = list(request.target)

            model_service.retrain_model(
                model_id=model_id,
                hyperparameters=hyperparameters,
                features=features,
                target=target,
            )
            logger.info(f"Переобучение модели завершено. ID модели: {model_id}")
            return ml_model_service_pb2.RetrainModelResponse(
                message=f"Модель с ID '{model_id}' успешно переобучена"
            )
        except TrainedModelNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_model_service_pb2.RetrainModelResponse()
        except ModelServiceError as e:
            logger.error(f"Ошибка при переобучении модели с ID '{model_id}': {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_model_service_pb2.RetrainModelResponse()


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