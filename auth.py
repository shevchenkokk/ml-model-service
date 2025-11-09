from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError

from database import get_user_from_database
import logging
logging.getLogger('passlib').setLevel(logging.ERROR)

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "default_key")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Проверяет, соответствует ли введённый пользователем пароль хэшу
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_user(username: str) -> Optional[User]:
    """
    Получает пользователя из БД и возвращает его в виде Pydantic
    """
    user_info = get_user_from_database(username)
    if user_info:
        return User(**user_info)
    return None


def create_access_token(data: dict):
    """
    Создает JWT-токен
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Декодирует и валидирует JWT-токен, возвращая информацию о пользователе
    """
    creds_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Некорректный или истёкший токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise creds_exception
    except JWTError:
        raise creds_exception

    user = get_user(username)
    if user is None:
        raise creds_exception
    return user