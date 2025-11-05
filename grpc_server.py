import grpc
from concurrent import futures
import logging
import uuid
import joblib
from pathlib import Path

from generated import (
    ml_model_service_pb2,
    ml_model_service_pb2_grpc
)

from database import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "Логистическая регрессия": "logistic_regression",
    "Случайный лес": "random_forest",
    "Градиентный бустинг (LightGBM)": "lightgbm"
}


class MLModelService(ml_model_service_pb2_grpc.MLModelServiceServicer):
    def GetAvailableModels(self, request, context):
        ...


    def TrainModel(self, request, context):
        ...


    def GetTrainedModels(self, request, context):
        ...

    
    def Predict(self, request, context):
        ...


    def DeleteModel(self, request, context):
        ...


    def RetrainModel(self, request, context):
        ...


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
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    logger.info(f"Создана папка '{TRAINED_MODELS_DIR}' для хранения обученных моделей")
    serve()