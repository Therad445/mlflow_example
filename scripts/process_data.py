import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, RANDOM_STATE, TEST_SIZE, DATA_DIR
from utils import get_logger, load_params

STAGE_NAME = "process_data"


def process_data() -> dict:
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info("Начали скачивать данные")
    dataset = load_dataset(DATASET_NAME)
    logger.info("Успешно скачали данные!")

    df = dataset["train"].to_pandas()

    features = params["features"]
    train_size_cfg = int(params.get("train_size", -1))

    target_column = "income"
    X = df[features]
    y = df[target_column]

    all_cat_features = [
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ]

    # сохраняем порядок фичей как в YAML
    cat_features = [c for c in features if c in all_cat_features]
    num_features = [c for c in features if c not in all_cat_features]

    logger.info(f"Используемые фичи: {features}")
    logger.info(f"Числовые: {num_features}")
    logger.info(f"Категориальные: {cat_features}")

    # числовые
    X_num = X[num_features].to_numpy() if num_features else np.empty((len(X), 0))

    # категориальные
    if cat_features:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X[cat_features])
    else:
        X_cat = np.empty((len(X), 0))

    X_transformed = np.hstack([X_num, X_cat]).astype(float)
    y_transformed = (y == ">50K").astype(int).to_numpy()

    # фиксируем тест одинаковым
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y_transformed,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_transformed,
    )

    # уменьшаем train
    if train_size_cfg is not None and train_size_cfg > 0 and train_size_cfg < len(y_train):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=train_size_cfg,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

    logger.info(f"Размер тренировочного датасета: {len(y_train)}")
    logger.info(f"Размер тестового датасета: {len(y_test)}")

    logger.info("Начали сохранять датасеты")
    os.makedirs(DATA_DIR, exist_ok=True)

    pd.DataFrame(X_train).to_csv(DATASET_PATH_PATTERN.format(split_name="X_train"), index=False)
    pd.DataFrame(X_test).to_csv(DATASET_PATH_PATTERN.format(split_name="X_test"), index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(
        DATASET_PATH_PATTERN.format(split_name="y_train"), index=False
    )
    pd.DataFrame(y_test, columns=["target"]).to_csv(
        DATASET_PATH_PATTERN.format(split_name="y_test"), index=False
    )

    logger.info("Успешно сохранили датасеты!")

    return {
        "features": features,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }


if __name__ == "__main__":
    process_data()
