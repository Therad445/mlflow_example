from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
PARAMS_DIR = PROJECT_ROOT / "params"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASET_PATH_PATTERN = str(DATA_DIR / "{split_name}.csv")
DATASET_NAME = "scikit-learn/adult-census-income"

MODEL_FILEPATH = PROJECT_ROOT / "model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.3

MLFLOW_TRACKING_URI = "http://158.160.2.37:5000"
EXPERIMENT_NAME = "homework_islamov"