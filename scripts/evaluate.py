import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, ARTIFACTS_DIR
from utils import get_logger, load_params

STAGE_NAME = "evaluate"


def evaluate(model_path: str | None = None) -> tuple[dict, str]:
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    threshold = float(params.get("threshold", 0.5))

    logger.info("Считываем датасеты")
    X_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="X_test"))
    y_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="y_test"))["target"].to_numpy()

    mp = model_path or str(MODEL_FILEPATH)

    logger.info(f"Загружаем обученную модель: {mp}")
    if not os.path.exists(mp):
        raise FileNotFoundError("Не нашли файл с моделью. Запусти train перед evaluate.")
    model = load(mp)

    # scores для ROC-AUC / PR-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        # нормализуем в [0,1], чтобы threshold имел смысл и метрики не падали
        y_score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    else:
        raise TypeError("Model has neither predict_proba nor decision_function")

    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
    }

    logger.info(f"Метрики: {metrics}")

    # Артефакты
    out_dir = Path(ARTIFACTS_DIR) / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "classification_report.txt"
    cm_path = out_dir / "confusion_matrix.csv"
    errors_path = out_dir / "errors.csv"

    report = classification_report(y_test, y_pred, digits=4)
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    # index_label — чтобы в MLflow/просмотрщиках не выглядело как “строки пропали”
    df_cm.to_csv(cm_path, index=True, index_label="true\\pred")

    # CSV с ошибками (только ошибочные строки)
    err_mask = (y_pred != y_test)
    if np.any(err_mask):
        err_df = X_test.loc[err_mask].copy()
        err_df["y_true"] = y_test[err_mask]
        err_df["y_pred"] = y_pred[err_mask]
        err_df["y_score"] = y_score[err_mask]
        err_df.to_csv(errors_path, index=False)
    else:
        pd.DataFrame(columns=list(X_test.columns) + ["y_true", "y_pred", "y_score"]).to_csv(
            errors_path, index=False
        )

    return metrics, str(out_dir)


if __name__ == "__main__":
    evaluate()