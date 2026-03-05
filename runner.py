from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from joblib import load as joblib_load

from constants import DATA_DIR, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from scripts import evaluate, process_data, train

def log_dataset_artifacts() -> None:
    data_dir = Path(DATA_DIR)

    train_files = ["X_train.csv", "y_train.csv"]
    test_files = ["X_test.csv", "y_test.csv"]

    logged = 0

    for name in train_files:
        p = data_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Dataset artifact not found: {p}")
        mlflow.log_artifact(str(p), artifact_path="datasets/train")
        logged += 1

    for name in test_files:
        p = data_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Dataset artifact not found: {p}")
        mlflow.log_artifact(str(p), artifact_path="datasets/test")
        logged += 1

    mlflow.set_tag("has_dataset_artifacts", "true")
    mlflow.log_param("dataset_artifacts_logged_files", logged)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        data_info = process_data()
        log_dataset_artifacts()
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


        # register the best artifact with metadata
        try:
            from scripts.register_model import register_model_to_registry

            registry_params = {
                "data__train_size": data_info["train_size"],
                "data__test_size": data_info["test_size"],
                "data__features": ",".join(data_info["features"]),
                "model_type": model_info["model_type"],
                **{f"model__{k}": v for k, v in model_info["model_params"].items()},
            }

            dataset_ref = f"mlflow:{MLFLOW_TRACKING_URI} exp={EXPERIMENT_NAME} run={mlflow.active_run().info.run_id} artifact=dataset/"
            code_ref = f"git:unknown (fill later)"

            reg = register_model_to_registry(
                metrics=metrics,
                params=registry_params,
                dataset_ref=dataset_ref,
                code_ref=code_ref,
            )
            print("Registered in registry:", reg)
        except Exception as e:
            print("Registry registration skipped:", e)

if __name__ == "__main__":
    main()