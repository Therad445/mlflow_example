from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from constants import DATA_DIR, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from scripts import evaluate, process_data, train


def log_dataset_artifacts():
    data_dir = Path(DATA_DIR)
    files = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]

    for name in files:
        p = data_dir / name
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="dataset")


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        data_info = process_data()
        log_dataset_artifacts()

        model, model_info = train()

        run_name = (
            f"{model_info['model_type']}_"
            f"ts{data_info['train_size']}_"
            f"f{len(data_info['features'])}_"
            f"{ts}"
        )
        mlflow.set_tag("mlflow.runName", run_name)

        # params
        mlflow.log_param("data__train_size", data_info["train_size"])
        mlflow.log_param("data__test_size", data_info["test_size"])
        mlflow.log_param("data__features", ",".join(data_info["features"]))

        mlflow.log_param("model_type", model_info["model_type"])
        mlflow.log_params({f"model__{k}": v for k, v in model_info["model_params"].items()})

        # metrics + artifacts
        metrics, artifact_paths = evaluate(model=model)
        mlflow.log_metrics(metrics)

        for p in artifact_paths:
            mlflow.log_artifact(p, artifact_path="artifacts")

        # model
        if model_info["model_type"].lower() == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()