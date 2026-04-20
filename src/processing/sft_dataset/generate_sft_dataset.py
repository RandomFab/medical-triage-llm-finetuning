import yaml

from config.logger import logger
from config.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, SFT_DATASET_DIR
from src.processing.anonymisation import anonymize_text
from src.processing.utils_cleaning import collect_balanced_samples


if __name__ == "__main__":
    """
    Generate a balanced SFT dataset by sampling from multiple medical datasets.

    Reads parameters from params.yaml (sft.target_samples, sft.random_state,
    sft.source_datasets). Outputs a single Parquet to
    data/processed/sft_dataset/sft_dataset.parquet (tracked by DVC).
    """
    with (PROJECT_ROOT / "params.yaml").open() as f:
        params = yaml.safe_load(f)["sft"]

    target_samples: int = params["target_samples"]
    random_state: int = params["random_state"]
    parquet_files: list[str] = params["source_datasets"]

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

    logger.info("=" * 60)
    logger.info(f"Final dataset size: {len(sft_dataset)} rows (Target: {target_samples})")

    output_path = SFT_DATASET_DIR / "sft_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_dataset.to_parquet(output_path, index=False)
    logger.info(f"Successfully saved {len(sft_dataset)} samples to {output_path}")
    logger.info("=" * 60)
