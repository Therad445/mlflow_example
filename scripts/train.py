import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = "train"


def _make_model(model_type: str, model_params: dict):
    mt = model_type.lower()

    if mt == "logistic_regression":
        return LogisticRegression(random_state=RANDOM_STATE, **model_params)

    if mt == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **model_params)

    if mt == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **model_params)

    if mt == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **model_params)

    if mt == "xgboost":
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **model_params)

    raise ValueError(
        f"Unknown model_type='{model_type}'. Use one of: "
        "logistic_regression, decision_tree, random_forest, gradient_boosting, xgboost"
    )


def train() -> tuple[object, dict]:
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
    dump(model, MODEL_FILEPATH)
    logger.info("Успешно!")

    meta = {"model_type": model_type, "model_params": model_params}
    return model, meta


if __name__ == "__main__":
    train()
