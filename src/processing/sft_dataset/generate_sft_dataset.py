from pathlib import Path

import pandas as pd
import yaml

from config.logger import logger
from config.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, SFT_DATASET_DIR


def extract_samples(
    parquet_file_path: Path, sample: int, random_state: int
) -> tuple[pd.DataFrame, int]:
    """
    Extract samples from a Parquet file, adapting to dataset size.

    Args:
        parquet_file_path: Absolute path to the parquet file.
        sample: Target number of samples to extract.
        random_state: Seed used by pandas.DataFrame.sample for reproducibility.

    Returns:
        Tuple of (sampled_dataframe, actual_number_of_samples_returned).
        If the dataset has fewer rows than requested, returns all available rows.
    """
    logger.info(f"Reading parquet file: {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)
    logger.info(f"Successfully loaded {len(df)} rows from {parquet_file_path}")

    if len(df) < sample:
        logger.warning(
            f"File {parquet_file_path} has only {len(df)} rows (requested: {sample}). "
            f"Returning all {len(df)} available rows."
        )
        return df, len(df)

    logger.info(f"Sampling {sample} rows from {parquet_file_path}")
    sampled_df = df.sample(n=sample, random_state=random_state)
    return sampled_df, sample


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

    sft_dataset = pd.DataFrame(columns=["question", "answer"])
    total_samples_collected = 0

    for idx, parquet_file in enumerate(parquet_files, 1):
        remaining_datasets = len(parquet_files) - idx + 1
        remaining_quota = target_samples - total_samples_collected
        to_sample = remaining_quota // remaining_datasets

        logger.info(f"[{idx}/{len(parquet_files)}] Processing {parquet_file}")
        logger.info(f"  Target samples: {to_sample} | Remaining quota: {remaining_quota}")

        sampled_df, nb_of_sample = extract_samples(
            PROCESSED_DATA_DIR / parquet_file,
            sample=to_sample,
            random_state=random_state,
        )
        total_samples_collected += nb_of_sample
        sft_dataset = pd.concat([sft_dataset, sampled_df], ignore_index=True)
        logger.info(
            f"  Added {nb_of_sample} samples | Total in dataset: {len(sft_dataset)}/{target_samples}"
        )

    logger.info("=" * 60)
    logger.info(f"Final dataset size: {len(sft_dataset)} rows (Target: {target_samples})")

    output_path = SFT_DATASET_DIR / "sft_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_dataset.to_parquet(output_path, index=False)
    logger.info(f"Successfully saved {len(sft_dataset)} samples to {output_path}")
    logger.info("=" * 60)
