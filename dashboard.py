import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(
    page_title="Управление ML-моделями",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"


# ------ ручки для обращения к API ------
def get_available_models():
    """Получает список классов моделей, доступных для обучения"""
    try:
        response = requests.get(f"{API_URL}/models")
        response.raise_for_status()
        return response.json()["available_models"]
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при загрузке списка доступных моделей: {e}")
        return {}


def get_trained_models(token):
    """Получает список всех обученных моделей"""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{API_URL}/trained-models", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при загрузке списка обученных моделей: {e}")
        return []


def train_model(token, model_name, hyperparameters, features, target):
    """Отправляет запрос на обучение модели"""
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "features": features,
        "target": target,
    }
    try:
        response = requests.post(f"{API_URL}/train", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при обучении модели: {e}")
        st.json(e.response.json())
        return None


def predict(token, model_id, features):
    """Отправляет запрос на получений предсказания"""
    headers = {"Authorization": f"Bearer {token}"}
    data = {"features": features}
    try:
        response = requests.post(f"{API_URL}/predict/{model_id}", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при получении предсказания: {e}")
        st.json(e.response.json())
        return None


def delete_model(token, model_id):
    """Отправляет запрос на удаление модели"""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.delete(f"{API_URL}/trained-models/{model_id}", headers=headers)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при удалении модели: {e}")
        return False


def retrain_model(token, model_id, hyperparameters, features, target):
    """Отправляет запрос на переобучение модели"""
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "hyperparameters": hyperparameters,
        "features": features,
        "target": target,
    }
    try:
        response = requests.put(f"{API_URL}/retrain/{model_id}", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при переобучении модели: {e}")
        st.json(e.response.json())
        return None


def register(username, password):
    """Отправляет запрос на регистрацию пользователя"""
    data = {
        "username": username,
        "password": password
    }
    try:
        response = requests.post(f"{API_URL}/register", json=data)
        response.raise_for_status()
        st.sidebar.success(f"Пользователь '{username}' успешно зарегистрирован!")
        return True
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Ошибка регистрации: {e.response.json().get('detail', 'Неизвестная ошибка')}")
        return False


def login(username, password):
    """Отправляет запрос на получение JWT-токена"""
    data = {
        "username": username,
        "password": password
    }
    try:
        response = requests.post(f"{API_URL}/token", data=data)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Ошибка входа: {e.response.json().get('detail', 'Неизвестная ошибка')}")
        return None


# ------ дашборд ------
st.title("Управление ML-моделями")
if 'token' not in st.session_state:
    st.session_state.token = None

with st.sidebar:
    st.header("Аутентификация")
    if st.session_state.token is None:
        st.subheader("Вход")
        username = st.text_input("Имя пользователя", value="user1")
        password = st.text_input("Пароль", type="password", value="password123")
        if st.button("Войти"):
            token = login(username, password)
            if token:
                st.session_state.token = token
                st.rerun()
        
        with st.expander("Регистрация"):
            reg_username = st.text_input("Новое имя пользователя", key="reg_user")
            reg_password = st.text_input("Новый пароль", type="password", key="reg_pass")
            if st.button("Зарегистрироваться"):
                if reg_username and reg_password:
                    register(reg_username, reg_password)
                else:
                    st.warning("Заполните все поля для регистрации")
    else:
        st.success("Вы вошли в систему.")
        if st.button("Выйти"):
            st.session_state.token = None
            st.rerun()

if st.session_state.token:
    # ------ отображение обученных моделей ------
    st.header("Обученные модели")
    trained_models = get_trained_models(st.session_state.token)
    if trained_models:
        df = pd.DataFrame(trained_models)
        st.dataframe(df)

        st.header("Переобучение существующей модели")
        st.write("Заполните все поля и загрузите файл для переобучения")

        model_ids_retrain = df["id"].tolist()
        model_id_to_retrain = st.selectbox(
            "Выберите модель для переобучения",
            options=[""] + model_ids_retrain,
            key="retrain_id_select"
        )
        st.text_area(
            "Новые гиперпараметры (в формате JSON)",
            key="retrain_hyperparams",
            value='{"n_estimators": 30}'
        )
        uploaded_retrain_file = st.file_uploader(
            "Загрузите CSV-файл с данными для переобучения",
            type="csv",
            key="retrain_file"
        )

        if uploaded_retrain_file is not None:
            try:
                df_retrain = pd.read_csv(uploaded_retrain_file)
                st.write("Предпросмотр данных для переобучения:")
                st.dataframe(df_retrain.head())

                target_column_retrain = st.selectbox(
                    "Выберите целевую переменную (таргет) для переобучения",
                    options=df_retrain.columns,
                    key="retrain_target_select",
                    index=len(df_retrain.columns) - 1
                )

                if st.button("Начать переобучение"):
                    with st.spinner("Модель переобучается..."):
                        hyperparams_str = st.session_state.retrain_hyperparams
                        hyperparams = json.loads(hyperparams_str)
                        
                        features_df = df_retrain.drop(columns=[target_column_retrain])
                        target_series = df_retrain[target_column_retrain]

                        features = features_df.values.tolist()
                        target = target_series.values.tolist()
                        
                        result = retrain_model(st.session_state.token, model_id_to_retrain, hyperparams, features, target)

                    if result:
                        st.session_state.retrain_success = result["message"]
                        st.rerun()
            except Exception as e:
                st.error(f"Ошибка при подготовке данных для переобучения: {e}")

        if 'retrain_success' in st.session_state:
            st.success(st.session_state.retrain_success)
            del st.session_state.retrain_success
        
        st.header("Удаление модели")
        models_to_delete = df["id"].tolist()
        model_id_to_delete = st.selectbox("Выберите модель для удаления", options=[""] + models_to_delete)
        if model_id_to_delete and st.button("Удалить модель"):
            if delete_model(st.session_state.token, model_id_to_delete):
                st.session_state.delete_success = f"Модель {model_id_to_delete} успешно удалена"
                st.rerun()

        if 'delete_success' in st.session_state:
            st.success(st.session_state.delete_success)
            del st.session_state.delete_success
    else:
        st.info("Пока нет ни одной обученной модели... Попробуй что-нибудь обучить :)")

    # --- обучение новой модели ------
    st.header("Обучение новой модели")
    st.write("Заполните все поля и загрузите файл для обучения")

    available_models = get_available_models()
    model_for_training = st.selectbox("Выберите класс модели", options=list(available_models.keys()))

    st.text_area("Гиперпараметры (в формате JSON)", key="hyperparams", value='{"n_estimators": 10}')

    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл с данными для обучения",
        type="csv"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Предпросмотр данных:")
            st.dataframe(df.head())

            target_column = st.selectbox(
                "Выберите целевую переменную (таргет)",
                options=df.columns,
                index=len(df.columns) - 1
            )

            if st.button("Начать обучение"):
                with st.spinner("Модель обучается..."):
                    model_name = available_models[model_for_training]
                    hyperparams_str = st.session_state.hyperparams
                    hyperparams = json.loads(hyperparams_str)

                    features_df = df.drop(columns=[target_column])
                    target_series = df[target_column]

                    features = features_df.values.tolist()
                    target = target_series.values.tolist()
            
                    result = train_model(st.session_state.token, model_name, hyperparams, features, target)
                if result:
                    st.session_state.train_success = f"Модель успешно обучена! ID: {result['trained_model_id']}"
                    st.rerun()
        except Exception as e:
            st.error(f"Ошибка при подготовке данных для обучения: {e}")

    if 'train_success' in st.session_state:
        st.success(st.session_state.train_success)
        del st.session_state.train_success

    # ------ получение предсказаний ------
    if trained_models:
        st.header("Получение предсказаний")
        model_ids = [model["id"] for model in trained_models]
        selected_model_id = st.selectbox("Выберите ID обученной модели", options=model_ids)
        
        uploaded_predict_file = st.file_uploader(
            "Загрузите CSV-файл с данными для предсказания (без целевой переменной)",
            type="csv",
            key="predict_file"
        )
        if uploaded_predict_file is not None:
            try:
                df_predict = pd.read_csv(uploaded_predict_file)
                st.write("Предпросмотр данных для предсказания:")
                st.dataframe(df_predict.head())
                
                features = df_predict.values.tolist()

                if st.button("Получить предсказания"):
                    with st.spinner("Получение предсказаний..."):
                        result = predict(st.session_state.token, selected_model_id, features)

                    if result:
                        st.success("Предсказания получены:")
                        df_predict["preds"] = result["preds"]
                        st.dataframe(df_predict)
                        csv = df_predict.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать предсказания в CSV",
                            data=csv,
                            file_name=f"preds_{selected_model_id}.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Ошибка при подготовке данных для предсказания: {e}")
else:
    st.warning("Пожалуйста, войдите в систему, чтобы получить доступ к дашборду.")