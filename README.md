# Сервис для управления ML-моделями

**Автор:** *Шевченко Кирилл*

студент группы мИПИИ241, ВШЭ, Москва

## Описание проекта

Данный проект – веб-сервис, предоставляющий `REST` и `gRPC` API для обучения, сохранения и использования моделей машинного обучения, с интерактивным дашбордом для управления, написанным с использованием фреймворка `Streamlit`.

## Технологический стек

* Python
* FastAPI
* Streamlit
* gRPC
* Scikit-learn
* Lightgbm
* SQLite
* Poetry

## Инструкция по установке

Чтобы развернуть проект у себя на машине, необходимо:

1. Клонировать репозиторий с помощью команды: `git clone https://github.com/shevchenkokk/ml-model-service.git`
2. Перейти в директорию проекта: `cd ml-model-service`
3. Настроить окружение и установить зависимости с помощью Poetry: `poetry install`
4. Создать файл `.env` в корне проекта для хранения секретов и параметров интеграций.

### Переменные окружения

Пример минимального `.env`:

```bash
# базовые настройки
SECRET_KEY="change-me"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# S3 / MinIO для хранения обученных моделей
S3_ENABLED=true
S3_ENDPOINT="http://localhost:9000"
S3_REGION="us-east-1"
S3_ACCESS_KEY="minioadmin"
S3_SECRET_KEY="minioadmin"
S3_BUCKET="ml-model-service"
S3_MODELS_PREFIX="models/"
```

> Если `S3_ENABLED=false`, сервис продолжит работать только с локальной директорией `trained_models`.

## Инструкция по запуску (локально, без Docker)

**Для проверки работы REST API** нужно запустить два сервиса, каждый в отдельном терминале:

1. REST API сервер
```bash
poetry run uvicorn app.main:app --reload
```

* Сервер будет доступен по адресу `http://127.0.0.1:8000`, все REST-эндпоинты находятся по префиксу `/api`
* Swagger находится по адресу `http://127.0.0.1:8000/docs`

2. Интерактивный дашборд
```bash
poetry run streamlit run dashboard/dashboard.py
```

Дашборд будет доступен по адресу `http://localhost:8501`.

Данные для тестирования находятся в директории `data` в корне проекта.
Протестировать можно, используя csv-файл `train_data.csv` с данными для обучения и файл `test_data_for_prediction.csv` с данными для предсказаний.

**Для проверки работы gRPC** нужно запустить gRPC сервер:
```bash
poetry run python grpc/grpc_server.py
```

После запуска вы должны увидеть, что сервер успешно запущен и база данных инициализирована.

Далее можно выполнить в отдельном терминале команду:
```bash
poetry run python grpc/grpc_client.py
```

Эта команда выполнит полный цикл операций (получение списка доступных моделей, обучение модели, получение списка обученных моделей, получение предсказаний, удаление модели, получение списка обученных моделей) и выведет результаты в консоль.

## Аутентификация

Все основные эндпоинты `REST API` защищены. Для доступа к ним необходимо получить `JWT-токен`.

**При тестировании дашборда:**

1. Убедитесь, что REST API сервер и дашборд запущены
2. На боковой панели дашборда вы увидите форму аутентификации.

Вы можете создать нового пользователя, используя форму `Регистрация` на боковой панели. После успешной регистрации и входа вы получите доступ ко всей функциональности сервиса.

## Работа с S3 (MinIO) и DVC

Интеграция с S3/MinIO и DVC работает одинаково для всех вариантов развёртывания: локально, через `docker-compose` и в Kubernetes. Важно лишь, чтобы были заданы корректные переменные окружения (`S3_*`, `AWS_*`) и был доступен MinIO.

### Как это устроено в приложении

DVC-логика реализована в модуле `app/storage/dvc.py`:
- `save_dataset(...)` — сохраняет датасет в `data/processed/...`;
- `version_dataset(path)` — добавляет файл в DVC и пушит в удалённый MinIO-remote;
- `setup_dvc_remote()` — настраивает DVC remote на MinIO (использует переменные окружения S3).

При старте приложения (`app/main.py`) вызывается `setup_dvc_remote()`, поэтому при корректно настроенных переменных окружения DVC remote будет настроен автоматически, а датасеты будут версионироваться и отправляться в MinIO — как при локальном запуске, так и внутри контейнеров.

### Режим локального запуска (без Docker)

1. Поднимите MinIO:
   ```bash
   MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
   minio server /tmp/minio-data --console-address :9001
   ```
2. Создайте бакеты:
   - `ml-model-service` — для хранения обученных моделей;
   - `mlflow-artifacts` — для артефактов MLflow.
3. Убедитесь, что в `.env` выставлены значения, как описано в разделе «Переменные окружения».

В этом режиме приложение будет сохранять модели и датасеты в локальную файловую систему и одновременно версионировать их в MinIO через DVC.

## Запуск через Docker и docker-compose

Для удобства вся инфраструктура (API, дашборд, MinIO, MLflow, DVC-remote) упакована в Docker-образы и запускается через `docker-compose`.

### Сборка образов

```bash
docker-compose build
```

Это соберёт:
- `ml-model-service-api` — образ FastAPI-сервиса;
- `ml-model-service-dashboard` — образ Streamlit-дашборда.

### Запуск всего стека

```bash
docker-compose up -d
```

Что поднимается:
- MinIO (`minio`): порты `9000` (API) и `9001` (консоль);
- MinIO init (`minio-init`): создаёт бакеты `ml-model-service` и `mlflow-artifacts`;
- MLflow (`mlflow`): порт `5001` (проброс на внутренний `5000`);
- API (`api`): порт `8000`;
- Dashboard (`dashboard`): порт `8501`.

Переменные окружения для сервисов (S3, MLflow, JWT и т.п.) задаются внутри `docker-compose.yml`. При запуске:
- API и DVC используют MinIO как S3-совместимое хранилище для моделей и версионированных датасетов (`ml-model-service`);
- MLflow использует MinIO для хранения артефактов обучений (`mlflow-artifacts`).

### Проверка работы

После `docker-compose up -d`:

- Health API:
  ```bash
  curl http://localhost:8000/health
  ```
- Swagger UI: `http://localhost:8000/docs`
- Дашборд: `http://localhost:8501`
- MinIO Console: `http://localhost:9001` (логин/пароль `minioadmin/minioadmin`)
- MLflow UI: `http://localhost:5001`

Логи отдельных сервисов:
```bash
docker-compose logs api
docker-compose logs minio
docker-compose logs mlflow
docker-compose logs dashboard
```

Остановка и удаление ресурсов:
```bash
docker-compose down
docker-compose down -v  # с удалением volume-ов
```

## Трекинг обучения в MLflow

Интеграция с MLflow реализована в модуле `app/tracking/mlflow.py`. При запуске обучения:

- создаётся/выбирается эксперимент `ml-model-service`;
- логируются гиперпараметры и метрики;
- обученные модели и артефакты сохраняются в S3/MinIO (бакет `mlflow-artifacts`).

В `docker-compose.yml` MLflow поднимается автоматически, а необходимые переменные окружения (`MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, `AWS_*`) уже прописаны в разделах `mlflow` и `api`.

## Запуск в Kubernetes

Сервис может быть развёрнут в Kubernetes (в minikube). При этом также используется связка MinIO + DVC + MLflow — переменные окружения S3/MLflow пробрасываются в поды из `configmap.yaml` и `secret.yaml`.

- директория `k8s/` с манифестами:
  - `namespace.yaml` — namespace `ml-model-service`;
  - `configmap.yaml` — конфигурация S3/MLflow;
  - `secret.yaml` — секреты (ключи, пароли);
  - `minio-deployment.yaml` — MinIO + Service;
  - `minio-init-job.yaml` — Job инициализации бакетов;
  - `mlflow-deployment.yaml` — MLflow + Service;
  - `api-deployment.yaml` — API + Service (NodePort / ClusterIP);
  - `dashboard-deployment.yaml` — Streamlit-дэшборд + Service.
- скрипт `scripts/deploy-k8s.sh` — автоматический деплой в minikube.

### Требования

- установленный `minikube`;
- установленный `kubectl`;
- Docker (используется для сборки образа, который затем попадает в локальный registry minikube).

### Деплой в minikube

```bash
chmod +x scripts/deploy-k8s.sh
bash scripts/deploy-k8s.sh
```

Скрипт:
- проверяет наличие `minikube` и `kubectl`;
- запускает minikube (если он ещё не запущен);
- настраивает Docker на использование Docker daemon внутри minikube;
- собирает образ `ml-model-service:latest` из `Dockerfile`;
- применяет все манифесты из директории `k8s/` в правильном порядке;
- ждёт готовности MinIO и создаёт бакеты;
- поднимает MLflow, API и дашборд.

После успешного деплоя можно воспользоваться `kubectl port-forward` для доступа к сервисам:

```bash
# API + Swagger
kubectl port-forward -n ml-model-service svc/api 8000:8000

# Dashboard
kubectl port-forward -n ml-model-service svc/dashboard 8501:8501

# MinIO (API + Console)
kubectl port-forward -n ml-model-service svc/minio 9000:9000 9001:9001

# MLflow
kubectl port-forward -n ml-model-service svc/mlflow 5000:5000
```

Для удаления всех ресурсов:
```bash
kubectl delete namespace ml-model-service
```