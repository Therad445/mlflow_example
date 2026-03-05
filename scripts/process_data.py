import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, RANDOM_STATE, TEST_SIZE, DATA_DIR
from utils import get_logger, load_params

STAGE_NAME = "process_data"


def _build_ordered_df(
    features: list[str],
    num_features: list[str],
    X_num: np.ndarray,
    cat_features: list[str],
    X_cat: np.ndarray,
) -> pd.DataFrame:
    """
    Собираем DataFrame строго в порядке features (как в YAML),
    чтобы порядок колонок был стабильным и очевидным.
    """
    num_idx = {f: i for i, f in enumerate(num_features)}
    cat_idx = {f: i for i, f in enumerate(cat_features)}

    data = {}
    for f in features:
        if f in num_idx:
            data[f] = X_num[:, num_idx[f]]
        else:
            data[f] = X_cat[:, cat_idx[f]]
    return pd.DataFrame(data, columns=features)


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
    y = (df[target_column] == ">50K").astype(int)  # Series 0/1

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

    # порядок этих списков тоже соответствует YAML
    cat_features = [f for f in features if f in all_cat_features]
    num_features = [f for f in features if f not in all_cat_features]

    logger.info(f"Используемые фичи (YAML order): {features}")
    logger.info(f"Числовые: {num_features}")
    logger.info(f"Категориальные: {cat_features}")

    # 1) сначала делаем split на raw-данных (без leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2) опционально уменьшаем train (и только потом fit энкодера)
    if train_size_cfg > 0 and train_size_cfg < len(y_train):
        X_train_raw, _, y_train, _ = train_test_split(
            X_train_raw,
            y_train,
            train_size=train_size_cfg,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

    logger.info(f"Размер тренировочного датасета: {len(y_train)}")
    logger.info(f"Размер тестового датасета: {len(y_test)}")

    # 3) числовые
    X_train_num = X_train_raw[num_features].to_numpy(dtype=float) if num_features else np.empty((len(X_train_raw), 0))
    X_test_num = X_test_raw[num_features].to_numpy(dtype=float) if num_features else np.empty((len(X_test_raw), 0))

    # 4) категориальные (fit только на train)
    if cat_features:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = enc.fit_transform(X_train_raw[cat_features])
        X_test_cat = enc.transform(X_test_raw[cat_features])
    else:
        X_train_cat = np.empty((len(X_train_raw), 0))
        X_test_cat = np.empty((len(X_test_raw), 0))

    # 5) собираем итоговые матрицы строго в порядке YAML + сохраняем имена колонок
    X_train_out = _build_ordered_df(features, num_features, X_train_num, cat_features, X_train_cat)
    X_test_out = _build_ordered_df(features, num_features, X_test_num, cat_features, X_test_cat)

    logger.info("Начали сохранять датасеты")
    os.makedirs(DATA_DIR, exist_ok=True)

    X_train_out.to_csv(DATASET_PATH_PATTERN.format(split_name="X_train"), index=False)
    X_test_out.to_csv(DATASET_PATH_PATTERN.format(split_name="X_test"), index=False)
    pd.DataFrame({"target": y_train.to_numpy()}).to_csv(DATASET_PATH_PATTERN.format(split_name="y_train"), index=False)
    pd.DataFrame({"target": y_test.to_numpy()}).to_csv(DATASET_PATH_PATTERN.format(split_name="y_test"), index=False)

    logger.info("Успешно сохранили датасеты!")

    return {
        "features": features,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }


if __name__ == "__main__":
    process_data()