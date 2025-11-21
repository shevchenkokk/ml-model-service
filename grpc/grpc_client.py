import grpc
from generated import (
    ml_model_service_pb2,
    ml_model_service_pb2_grpc
)
from google.protobuf.empty_pb2 import Empty


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = ml_model_service_pb2_grpc.MLModelServiceStub(channel)

        try:
            print("GetAvailableModels:")
            available_models = stub.GetAvailableModels(Empty())
            print(f"Доступные модели: {dict(available_models.models)}")

            print("TrainModel:")
            train_req = ml_model_service_pb2.TrainModelRequest(
                model_name="random_forest",
                target=[0, 1]
            )
            train_req.hyperparameters.update({"n_estimators": 10})
            train_req.features.extend([
                ml_model_service_pb2.Feature(values=val) for val in [[1, 1], [5, 7]]
            ])
            train_resp = stub.TrainModel(train_req)
            print(train_resp.message)

            print("GetTrainedModels:")
            trained_models = stub.GetTrainedModels(Empty())
            print(f"Найдено {len(trained_models.models)} обученных моделей")
            for model in trained_models.models:
                print(f"ID: {model.id}, name: {model.model_name}")
            
            print("Predict:")
            predict_req = ml_model_service_pb2.ModelPredictRequest(model_id=train_resp.trained_model_id)
            predict_req.features.extend([
                ml_model_service_pb2.Feature(values=val) for val in [[2, 2], [8, 10]]
            ])
            predict_resp = stub.Predict(predict_req)
            print(f"Полученные предсказания: {list(predict_resp.preds)}")

            print("Retrain:")
            retrain_req = ml_model_service_pb2.RetrainModelRequest(
                model_id=train_resp.trained_model_id,
                target=[1, 0]
            )
            retrain_req.hyperparameters.update({"n_estimators": 30})
            retrain_req.features.extend([
                ml_model_service_pb2.Feature(values=val) for val in [[3, 1], [4, 9]]
            ])
            retrain_resp = stub.RetrainModel(retrain_req)
            print(retrain_resp.message)

            print("Delete:")
            delete_req = ml_model_service_pb2.DeleteModelRequest(model_id=train_resp.trained_model_id)
            delete_resp = stub.DeleteModel(delete_req)
            print(delete_resp.message)

            print("GetTrainedModels:")
            trained_models = stub.GetTrainedModels(Empty())
            print(f"Осталось {len(trained_models.models)} обученных моделей")

        except grpc.RpcError as e:
            print(f"Произошла ошибка gRPC: [{e.code()}] {e.details()}")


if __name__ == "__main__":
    run()