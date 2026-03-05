from functools import partial

import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = "train"

MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": partial(XGBClassifier, eval_metric="logloss"),
}


def _make_model(model_type: str, model_params: dict):
    mt = model_type.lower()
    if mt not in MODELS:
        raise ValueError(
            f"Unknown model_type='{model_type}'. Use one of: {', '.join(MODELS.keys())}"
        )
    return MODELS[mt](random_state=RANDOM_STATE, **model_params)


def train() -> dict:
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    model_type = params.get("model_type", "logistic_regression")
    model_params = params.get("model_params", {})

    logger.info("Начали считывать датасеты")
    X_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="X_train"))
    y_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="y_train"))["target"]
    logger.info("Успешно считали датасеты!")

    logger.info(f"Создаём модель: {model_type}")
    logger.info(f"Параметры модели: {model_params}")
    model = _make_model(model_type=model_type, model_params=model_params)

    logger.info("Обучаем модель")
    model.fit(X_train, y_train)

    logger.info("Сохраняем модель (joblib)")
    dump(model, str(MODEL_FILEPATH))
    logger.info(f"Успешно! model_path={MODEL_FILEPATH}")

    return {
        "model_type": model_type,
        "model_params": model_params,
        "model_path": str(MODEL_FILEPATH),
    }


if __name__ == "__main__":
    train()