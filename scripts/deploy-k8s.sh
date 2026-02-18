#!/usr/bin/env bash
# скрипт автоматического развертывания приложения в Kubernetes (minikube)
# выполняет: запуск кластера, сборку образа, применение манифестов в правильном порядке

# настройки безопасности bash:
# -e: остановка при любой ошибке
# -u: ошибка при использовании неопределенных переменных
# -o pipefail: ошибка в пайпе останавливает весь пайп
set -euo pipefail

# SCRIPT_DIR - директория, где находится этот скрипт
# ROOT_DIR - корень проекта (на уровень выше scripts/)
# K8S_DIR - директория с Kubernetes манифестами
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
K8S_DIR="${ROOT_DIR}/k8s"

echo "Развертывание приложения в Kubernetes (minikube)"

# проверяем, что minikube и kubectl установлены и доступны в PATH
if ! command -v minikube >/dev/null 2>&1; then
  echo "minikube не установлен. Установите: https://minikube.sigs.k8s.io/docs/start/"
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl не установлен. Установите: https://kubernetes.io/docs/tasks/tools/"
  exit 1
fi

# запуск minikube кластера
# проверяем статус minikube
# если не запущен - запускаем с параметрами:
# --memory=4096: выделяем 4GB RAM
# --cpus=2: выделяем 2 CPU ядра
# если запущен - пропускаем этот шаг
if ! minikube status >/dev/null 2>&1; then
  echo "Запуск minikube кластера..."
  minikube start --memory=4096 --cpus=2
else
  echo "minikube уже запущен"
fi

# настройка Docker для работы с minikube
# minikube имеет свой внутренний Docker registry
# minikube docker-env выводит команды для настройки переменных окружения
echo "Настройка Docker для minikube..."
eval $(minikube docker-env)

# сборка Docker образа приложения
echo "Сборка Docker образа..."
cd "${ROOT_DIR}"
docker build -t ml-model-service:latest .

# применение манифестов Kubernetes
echo "Применение манифестов Kubernetes..."
kubectl apply -f "${K8S_DIR}/namespace.yaml"
kubectl apply -f "${K8S_DIR}/configmap.yaml"
kubectl apply -f "${K8S_DIR}/secret.yaml"
kubectl apply -f "${K8S_DIR}/minio-deployment.yaml"

# ожидание готовности MinIO
echo "Ожидание готовности MinIO..."
kubectl wait --for=condition=available --timeout=120s deployment/minio -n ml-model-service || true

# инициализация бакетов MinIO
echo "Инициализация бакетов MinIO..."
kubectl apply -f "${K8S_DIR}/minio-init-job.yaml"
kubectl wait --for=condition=complete --timeout=60s job/minio-init -n ml-model-service || true

# запуск зависимых сервисов
echo "Запуск MLflow..."
kubectl apply -f "${K8S_DIR}/mlflow-deployment.yaml"

echo "Запуск API..."
kubectl apply -f "${K8S_DIR}/api-deployment.yaml"

echo "Запуск Dashboard..."
kubectl apply -f "${K8S_DIR}/dashboard-deployment.yaml"

echo "Ожидание готовности сервисов..."
kubectl wait --for=condition=available --timeout=300s deployment/mlflow -n ml-model-service
kubectl wait --for=condition=available --timeout=300s deployment/api -n ml-model-service
kubectl wait --for=condition=available --timeout=300s deployment/dashboard -n ml-model-service

# вывод информации о развертывании
echo ""
echo "Развертывание завершено!"
echo ""
echo "Статус подов:"
kubectl get pods -n ml-model-service