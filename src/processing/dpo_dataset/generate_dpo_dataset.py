import yaml

from config.logger import logger
from config.paths import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    DPO_RAW_DATASET_PATH,
    DPO_TEST_DATASET_PATH,
    DPO_TRAIN_DATASET_PATH,
    DPO_VAL_DATASET_PATH,
)
from src.processing.anonymisation import anonymize_text
from src.processing.utils_cleaning import (
    add_token_counts,
    collect_balanced_samples,
    split_dataset,
    save_cleaned_data_local
)


def main():
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
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        params = yaml.safe_load(f)["dpo"]

    target_samples: int = params["target_samples"]
    random_state: int = params["random_state"]
    parquet_files: list[str] = params["source_datasets"]
    val_size: float = params.get("val_size", 0.2)
    test_size: float = params.get("test_size", 0.1)

    logger.info("=" * 60)
    logger.info("Starting DPO dataset generation process")
    logger.info(
        f"Target: {target_samples} balanced samples from {len(parquet_files)} datasets"
    )
    logger.info("=" * 60)

    dpo_dataset = collect_balanced_samples(
        parquet_files=parquet_files,
        base_dir=PROCESSED_DATA_DIR,
        target_samples=target_samples,
        random_state=random_state,
    )
    # Retrait du traitement presidio car les noms des maladies sont identifiés comme des entités à anonymiser, ce qui pose problème pour la qualité du dataset DPO (ex: "Diabète de type 2" devient "Diabète de type [PERSON]") et rend les questions/réponses incohérentes.
    # columns_to_anonymize = ["question", "chosen", "rejected"]
    # for col in columns_to_anonymize:
    #     dpo_dataset[col] = dpo_dataset[col].map(anonymize_text)

    dpo_dataset = add_token_counts(
        dpo_dataset, columns=["question", "chosen", "rejected"]
    )

    logger.info("=" * 60)
    logger.info(
        f"Final dataset size: {len(dpo_dataset)} rows (Target: {target_samples})"
    )

    save_cleaned_data_local(dpo_dataset, DPO_RAW_DATASET_PATH)

    logger.info(f"Successfully saved {len(dpo_dataset)} samples to {DPO_RAW_DATASET_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
