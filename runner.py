from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from joblib import load as joblib_load

from constants import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from scripts import evaluate, process_data, train


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        data_info = process_data()
        model_info = train()

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

        # metrics + artifacts (логируем директорию целиком)
        metrics, artifacts_dir = evaluate(model_path=model_info["model_path"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(artifacts_dir, artifact_path="artifacts/evaluate")

        # model (как MLflow model, не как artifact)
        model = joblib_load(model_info["model_path"])
        if model_info["model_type"].lower() == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()