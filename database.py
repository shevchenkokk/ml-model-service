import sqlite3
import json
from pathlib import Path
from typing import Optional, Any
from passlib.context import CryptContext

DB_FILE = "ml_model_service.db"
TRAINED_MODELS_DIR = Path("trained_models")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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


def get_model_from_database(model_id: str) -> Optional[dict[str, Any]]:
    """
    Получает информацию о модели по ее ID.
    """
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT id, model_name, hyperparameters, model_path FROM trained_models WHERE id = ?",
        (model_id,)
    )
    row = cur.fetchone()
    con.close()

    if row is None:
        return None

    row_to_return = dict(row)
    row_to_return["hyperparameters"] = json.loads(row_to_return["hyperparameters"])

    return dict(row_to_return)


def get_trained_models_from_database():
    """
    Получает список всех обученных моделей.
    """
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT id, model_name, hyperparameters, model_path FROM trained_models"
    )
    rows = cur.fetchall()
    con.close()

    rows_to_return = []
    for row in rows:
        row_to_return = dict(row)
        row_to_return["hyperparameters"] = json.loads(row_to_return["hyperparameters"])
        rows_to_return.append(row_to_return)
    return rows_to_return


def delete_model_from_database(model_id: str) -> int:
    """
    Удаляет запись о модели по её ID.
    """
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute(
        "DELETE FROM trained_models WHERE id = ?",
        (model_id,)
    )
    con.commit()
    deleted_rows_num = cur.rowcount
    con.close()
    return deleted_rows_num


def update_model_in_database(model_id: str, new_hyperparameters: dict) -> int:
    """
    Обновляет информацию о модели (гиперпараметры).
    """
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    hyperparameters_json = json.dumps(new_hyperparameters)
    cur.execute(
        "UPDATE trained_models SET hyperparameters = ? WHERE id = ?",
        (hyperparameters_json, model_id)
    )
    con.commit()
    updated_rows_num = cur.rowcount
    con.close()
    return updated_rows_num


def create_users_table():
    """
    Создает таблицу users, если она не существует
    """
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def create_user_in_database(username: str, password: str) -> dict:
    """
    Создает нового пользователя в БД. Хэширует пароль перед сохранением.
    Возвращает данные о созданном пользователе.
    """
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    hashed_password = pwd_context.hash(password)
    try:
        cur.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (username, hashed_password)
        )
        con.commit()
    except sqlite3.IntegrityError:
        # если пользователь с таким username уже существует
        con.close()
        raise ValueError(f"Пользователь с именем '{username}' уже существует")
    con.close()
    return {"username": username, "hashed_password": hashed_password}


def get_user_from_database(username: str) -> Optional[dict]:
    """
    Находит пользователя в БД по его имени и возвращает инфу по нему
    """
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    )
    row = cur.fetchone()
    con.close()

    if row is None:
        return None
    return dict(row)