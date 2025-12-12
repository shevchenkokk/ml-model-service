.PHONY: build-and-push test lint

# переменные для Docker Hub
DOCKER_USERNAME ?= your-username
IMAGE_NAME ?= ml-model-service
IMAGE_TAG ?= latest
DOCKER_IMAGE ?= $(DOCKER_USERNAME)/$(IMAGE_NAME):$(IMAGE_TAG)

# переменные для Poetry
POETRY ?= poetry

# сборка Docker образа и пуш в Docker Hub
build-and-push:
	@echo "Сборка Docker образа..."
	docker build -t $(DOCKER_IMAGE) .
	@echo "Пуш образа в Docker Hub..."
	docker image push $(DOCKER_IMAGE)
	@echo "Образ $(DOCKER_IMAGE) успешно загружен в Docker Hub"

# запуск тестов
test:
	@echo "Запуск тестов..."
	$(POETRY) run pytest tests/ -v

# запуск линтеров
lint:
	@echo "Запуск линтеров..."
	$(POETRY) run ruff check app dashboard tests

