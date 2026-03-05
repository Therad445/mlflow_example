import hashlib
import os
from pathlib import Path
import requests

from constants import MODEL_FILEPATH, EXPERIMENT_NAME

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8009")
MODEL_NAME = os.getenv("REGISTRY_MODEL_NAME", "adult_income_classifier")
OWNER = os.getenv("REGISTRY_OWNER", os.getenv("USER", "mlds_unknown"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register_model_to_registry(
    metrics: dict,
    params: dict,
    dataset_ref: str | None,
    code_ref: str | None,
    run_id: str | None = None,
    model_path: str = MODEL_FILEPATH,
):
    # Create model
    r = requests.get(f"{REGISTRY_URL}/v1/models", params={"query": MODEL_NAME}, timeout=20)
    r.raise_for_status()
    items = r.json()
    model_id = None
    for it in items:
        if it["name"] == MODEL_NAME:
            model_id = it["id"]
            break

    if model_id is None:
        r = requests.post(
            f"{REGISTRY_URL}/v1/models",
            json={"name": MODEL_NAME, "description": "registered from mlflow pipeline", "owner": OWNER, "tags": ["mlflow", "homework"]},
            timeout=20,
        )
        r.raise_for_status()
        model_id = r.json()["id"]

    # Presign upload
    run_id = os.getenv("MLFLOW_RUN_ID", "manual")
    rid = run_id or os.getenv("MLFLOW_RUN_ID") or "manual"
    obj_name = f"models/{MODEL_NAME}/runs/{rid}/model.joblib"
    r = requests.post(
        f"{REGISTRY_URL}/v1/artifacts/presign-upload",
        json={"object_name": obj_name, "content_type": "application/octet-stream"},
        timeout=20,
    )
    r.raise_for_status()
    put_url = r.json()["url"]
    headers = r.json().get("headers", {})

    # Upload bytes to MinIO
    with open(model_path, "rb") as f:
        up = requests.put(put_url, data=f, headers=headers, timeout=120)
        up.raise_for_status()

    # Commit artifact
    sha = sha256_file(model_path)
    size = Path(model_path).stat().st_size

    r = requests.post(
        f"{REGISTRY_URL}/v1/artifacts/commit",
        json={"object_name": obj_name, "sha256": sha, "size_bytes": size},
        timeout=20,
    )
    r.raise_for_status()
    artifact_id = r.json()["id"]

    # Create version with metadata
    payload = {
        "metrics": metrics,
        "params": params,
        "env": {"python": "3.12"},
        "dataset_ref": dataset_ref,
        "code_ref": code_ref,
        "artifact_id": artifact_id,
    }
    r = requests.post(f"{REGISTRY_URL}/v1/models/{model_id}/versions", json=payload, timeout=20)
    r.raise_for_status()
    version = r.json()

    return {"model_id": model_id, "version": version}


if __name__ == "__main__":
    res = register_model_to_registry(
        metrics={"roc_auc": 0.0},
        params={"experiment": EXPERIMENT_NAME},
        dataset_ref="local:data/",
        code_ref="local",
        run_id="manual",
    )
    print(res)