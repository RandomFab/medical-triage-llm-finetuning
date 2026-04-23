from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SFT_DATASET_DIR = PROCESSED_DATA_DIR / "sft_dataset"
DPO_DATASET_DIR = PROCESSED_DATA_DIR / "dpo_dataset"
SFT_TRAIN_DATASET_PATH = SFT_DATASET_DIR / "sft_train_dataset.parquet"
SFT_VAL_DATASET_PATH = SFT_DATASET_DIR / "sft_val_dataset.parquet"
SFT_TEST_DATASET_PATH = SFT_DATASET_DIR / "sft_test_dataset.parquet"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
ROOT_MODEL_DIR = PROJECT_ROOT / "models"
RAW_DATA_GCS_URL = "gs://p14-medical-data/raw_data"
