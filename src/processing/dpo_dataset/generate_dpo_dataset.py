import yaml

from config.logger import logger
from config.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, DPO_DATASET_DIR
from src.processing.anonymisation import anonymize_text
from src.processing.utils_cleaning import add_token_counts, collect_balanced_samples, split_dataset


if __name__ == "__main__":
    """
    Generate a balanced DPO dataset by sampling from multiple medical datasets.

    Reads parameters from params.yaml (dpo.target_samples, dpo.random_state,
    dpo.source_datasets, dpo.val_size, dpo.test_size). Outputs four Parquet files
    to data/processed/dpo_dataset/ (tracked by DVC):
      - dpo_dataset.parquet  : full dataset before splitting
      - dpo_train.parquet    : training split
      - dpo_val.parquet      : validation split
      - dpo_test.parquet     : test split
    """
    with (PROJECT_ROOT / "params.yaml").open() as f:
        params = yaml.safe_load(f)["dpo"]

    target_samples: int = params["target_samples"]
    random_state: int = params["random_state"]
    parquet_files: list[str] = params["source_datasets"]
    val_size: float = params.get("val_size", 0.2)
    test_size: float = params.get("test_size", 0.1)

    logger.info("=" * 60)
    logger.info("Starting DPO dataset generation process")
    logger.info(f"Target: {target_samples} balanced samples from {len(parquet_files)} datasets")
    logger.info("=" * 60)

    dpo_dataset = collect_balanced_samples(
        parquet_files=parquet_files,
        base_dir=PROCESSED_DATA_DIR,
        target_samples=target_samples,
        random_state=random_state,
    )

    columns_to_anonymize = ["question", "chosen", "rejected"]
    for col in columns_to_anonymize:
        dpo_dataset[col] = dpo_dataset[col].map(anonymize_text)

    dpo_dataset = add_token_counts(dpo_dataset, columns=["question", "chosen", "rejected"])

    logger.info("=" * 60)
    logger.info(f"Final dataset size: {len(dpo_dataset)} rows (Target: {target_samples})")

    output_path = DPO_DATASET_DIR / "dpo_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dpo_dataset.to_parquet(output_path, index=False)

    X_train, X_val, X_test= split_dataset(dpo_dataset, random_state=random_state, val_size=0.2, test_size=0.1)
    X_train.to_parquet(DPO_DATASET_DIR / "dpo_train.parquet", index=False)
    X_val.to_parquet(DPO_DATASET_DIR / "dpo_val.parquet", index=False)
    X_test.to_parquet(DPO_DATASET_DIR / "dpo_test.parquet", index=False)

    logger.info(f"Successfully saved {len(dpo_dataset)} samples to {output_path}")
    logger.info("=" * 60)
