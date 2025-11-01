import sqlite3
import json
from pathlib import Path

DB_FILE = "ml_model_service.db"
TRAINED_MODELS_DIR = Path("trained_models")


def init_database():
    """
    Инициализирует базу данных: создает файл и таблицу
    для хранения данных по обученным моделям, если она ещё не существует.
    """
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trained_models (
            id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            hyperparameters TEXT,
            model_path TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def add_model_to_database(
    model_id: str,
    model_name: str,
    hyperparameters: dict,
    model_path: Path
):
    """
    Добавляет запись о новой обученной модели в базу данных.
    """
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    hyperparameters_json = json.dumps(hyperparameters)
    cur.execute(
        "INSERT INTO trained_models (id, model_name, hyperparameters, model_path) VALUES (?, ?, ?, ?)",
        (model_id, model_name, hyperparameters_json, str(model_path))
    )
    con.commit()
    con.close()