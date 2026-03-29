from functools import partial

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import ALL_CATEGORICAL_FEATURES, DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
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


def _build_pipeline(model_type: str, model_params: dict, features: list[str]) -> Pipeline:
    categorical_features = [f for f in features if f in ALL_CATEGORICAL_FEATURES]
    numeric_features = [f for f in features if f not in ALL_CATEGORICAL_FEATURES]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0,
    )

    model = _make_model(model_type=model_type, model_params=model_params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def train() -> dict:
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    model_type = params.get("model_type", "logistic_regression")
    model_params = params.get("model_params", {})

    logger.info("Начали считывать датасеты")
    X_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="X_train"))
    y_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="y_train"))["target"]
    logger.info("Успешно считали датасеты!")

    features = list(X_train.columns)

    logger.info(f"Создаём модель: {model_type}")
    logger.info(f"Параметры модели: {model_params}")

    pipeline = _build_pipeline(
        model_type=model_type,
        model_params=model_params,
        features=features,
    )

    logger.info("Обучаем pipeline")
    pipeline.fit(X_train, y_train)

    logger.info("Сохраняем pipeline (joblib)")
    dump(pipeline, str(MODEL_FILEPATH))
    logger.info(f"Успешно! model_path={MODEL_FILEPATH}")

    return {
        "model_type": model_type,
        "model_params": model_params,
        "model_path": str(MODEL_FILEPATH),
    }


if __name__ == "__main__":
    train()