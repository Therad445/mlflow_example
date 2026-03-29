import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

    features: list[str] = params["features"]
    train_size_cfg = int(params.get("train_size", -1))

    target_column = "income"
    X = df[features].copy()
    y = (df[target_column] == ">50K").astype(int)

    logger.info(f"Используемые фичи (YAML order): {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if train_size_cfg > 0 and train_size_cfg < len(y_train):
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

    X_train.to_csv(DATASET_PATH_PATTERN.format(split_name="X_train"), index=False)
    X_test.to_csv(DATASET_PATH_PATTERN.format(split_name="X_test"), index=False)
    pd.DataFrame({"target": y_train.to_numpy()}).to_csv(
        DATASET_PATH_PATTERN.format(split_name="y_train"), index=False
    )
    pd.DataFrame({"target": y_test.to_numpy()}).to_csv(
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