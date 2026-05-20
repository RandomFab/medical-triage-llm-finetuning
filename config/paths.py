from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
ROOT_MODEL_DIR = PROJECT_ROOT / "models"

# === DPO dataset paths === 
DPO_DATASET_DIR = PROCESSED_DATA_DIR / "dpo_dataset"
DPO_RAW_DATASET_PATH = DPO_DATASET_DIR / "dpo_dataset.parquet"
DPO_TRAIN_DATASET_PATH = DPO_DATASET_DIR / "dpo_train.parquet"
DPO_VAL_DATASET_PATH = DPO_DATASET_DIR / "dpo_val.parquet"
DPO_TEST_DATASET_PATH = DPO_DATASET_DIR / "dpo_test.parquet"

# === SFT dataset paths ===
SFT_DATASET_DIR = PROCESSED_DATA_DIR / "sft_dataset"
SFT_RAW_DATASET_PATH = SFT_DATASET_DIR / "sft_dataset.parquet"
SFT_AUGMENTED_DATASET_PATH = SFT_DATASET_DIR / "sft_dataset_augmented.parquet"
SFT_TRAIN_DATASET_PATH = SFT_DATASET_DIR / "sft_train.parquet"
SFT_VAL_DATASET_PATH = SFT_DATASET_DIR / "sft_val.parquet"
SFT_TEST_DATASET_PATH = SFT_DATASET_DIR / "sft_test.parquet"


RAW_DATA_GCS_URL = os.environ.get("RAW_DATA_GCS_URL", "gs://p14-medical-data/raw_data")

GCS_MODEL_PATH = os.environ.get("GCS_MODEL_PATH", "gs://p14-medical-data/mlflow-artifacts")
LOCAL_MERGED_MODEL_PATH = ROOT_MODEL_DIR / "merged_model_for_deployment"
LOCAL_MERGED_SFT_MODEL_PATH = LOCAL_MERGED_MODEL_PATH / "merged_model_sft"
LOCAL_MERGED_DPO_MODEL_PATH = LOCAL_MERGED_MODEL_PATH / "merged_model_dpo"
GCS_MERGED_MODEL_PATH = os.environ.get("GCS_MERGED_MODEL_PATH", "gs://p14-medical-data/merged-model-for-deployment")
