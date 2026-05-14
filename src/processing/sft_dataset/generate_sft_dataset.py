import yaml

from config.logger import logger
from config.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, SFT_DATASET_DIR,SFT_TEST_DATASET_PATH, SFT_TRAIN_DATASET_PATH, SFT_VAL_DATASET_PATH
from src.processing.anonymisation import anonymize_text
from src.processing.utils_cleaning import add_token_counts, collect_balanced_samples, split_dataset

def main():
    """
    Generate a balanced SFT dataset by sampling from multiple medical datasets.

    Reads parameters from params.yaml (sft.target_samples, sft.random_state,
    sft.source_datasets, sft.val_size, sft.test_size). Outputs four Parquet files
    to data/processed/sft_dataset/ (tracked by DVC):
      - sft_dataset.parquet  : full dataset before splitting
      - sft_train.parquet    : training split
      - sft_val.parquet      : validation split
      - sft_test.parquet     : test split
    """
    with (PROJECT_ROOT / "params.yaml").open() as f:
        params = yaml.safe_load(f)["sft"]

    target_samples: int = params["target_samples"]
    random_state: int = params["random_state"]
    parquet_files: list[str] = params["source_datasets"]
    val_size: float = params.get("val_size", 0.2)
    test_size: float = params.get("test_size", 0.1)

    logger.info("=" * 60)
    logger.info("Starting SFT dataset generation process")
    logger.info(f"Target: {target_samples} balanced samples from {len(parquet_files)} datasets")
    logger.info("=" * 60)

    sft_dataset = collect_balanced_samples(
        parquet_files=parquet_files,
        base_dir=PROCESSED_DATA_DIR,
        target_samples=target_samples,
        random_state=random_state,
    )

    columns_to_anonymize = ["question", "answer"]
    for col in columns_to_anonymize:
        sft_dataset[col] = sft_dataset[col].map(anonymize_text)

    sft_dataset = add_token_counts(sft_dataset, columns=["question", "answer"])

    logger.info("=" * 60)
    logger.info(f"Final dataset size: {len(sft_dataset)} rows (Target: {target_samples})")

    output_path = SFT_DATASET_DIR / "sft_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_dataset.to_parquet(output_path, index=False)
    X_train, X_val, X_test= split_dataset(
        sft_dataset, 
        random_state=random_state, 
        val_size=val_size, 
        test_size=test_size
        )
    X_train.to_parquet(SFT_TRAIN_DATASET_PATH, index=False)
    X_val.to_parquet(SFT_VAL_DATASET_PATH, index=False)
    X_test.to_parquet(SFT_TEST_DATASET_PATH, index=False)


    logger.info(f"Successfully saved {len(sft_dataset)} samples to {output_path}")
    logger.info("=" * 60)
    
if __name__ == "__main__":
    main()