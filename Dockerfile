# используем базовый образ python 3.10
FROM python:3.10-slim

# устанавливаем системные зависимости (компиляторы, git для dvc, curl для healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# устанавливаем Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# настраиваем Poetry (не создаём виртуальное окружение, т.к. в Docker контейнер уже изолирован)
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# рабочая директория
WORKDIR /app

# копируем зависимости
COPY pyproject.toml poetry.lock* ./

# отключаем создание venv (устанавливаем в системный Python)
RUN poetry config virtualenvs.create false

# устанавливаем зависимости
RUN poetry install --only main && rm -rf $POETRY_CACHE_DIR

# копируем код приложения
COPY . .

# создаём необходимые директории
RUN mkdir -p trained_models data/raw data/processed

# открываем порт
EXPOSE 8000

# команда запуска (используем python напрямую, т.к. пакеты в системном Python)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

