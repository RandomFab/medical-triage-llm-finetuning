from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SFT_DATASET_DIR = PROCESSED_DATA_DIR / "sft_dataset"

RAW_DATA_GCS_URL = "gs://p14-medical-data/raw_data"
