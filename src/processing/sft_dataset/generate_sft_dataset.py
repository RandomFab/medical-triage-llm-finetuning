import pandas as pd

from config.logger import logger

BASE_URL = "gs://p14-medical-data/processed_data/"


def extract_1250_samples(parquet_file_path: str, sample: int = 1250) -> tuple[pd.DataFrame, int]:
    """
    Extract samples from a Parquet file, adapting to dataset size.

    Args:
        parquet_file_path: Relative path to the parquet file from BASE_URL
        sample: Target number of samples to extract (default: 1250)

    Returns:
        Tuple of (sampled_dataframe, actual_number_of_samples_returned)
        If the dataset has fewer rows than requested, returns all available rows.
    """
    logger.info(f"Reading parquet file: {parquet_file_path}")
    df = pd.read_parquet(f'{BASE_URL}{parquet_file_path}')
    logger.info(f"Successfully loaded {len(df)} rows from {parquet_file_path}")

    # Adapt sample size if dataset is too small
    actual_sample_size = min(sample, len(df))

    if len(df) < sample:
        logger.warning(
            f"File {parquet_file_path} has only {len(df)} rows (requested: {sample}). "
            f"Returning all {len(df)} available rows."
        )
        return df, len(df)

    logger.info(f"Sampling {actual_sample_size} rows from {parquet_file_path}")
    sampled_df = df.sample(n=actual_sample_size, random_state=42)
    logger.info(f"Successfully sampled {actual_sample_size} rows from {parquet_file_path}")

    return sampled_df, actual_sample_size



if __name__ == "__main__":
    """
    Generate a balanced SFT dataset by sampling from multiple medical datasets.

    The script dynamically distributes samples across 4 datasets to reach a target of 5000 rows.
    If a dataset has fewer rows than requested, it uses all available rows and redistributes
    the remaining quota to the next datasets.

    Output: Parquet file at gs://p14-medical-data/processed_data/sft_dataset/sft_dataset.parquet
    """
    logger.info("=" * 60)
    logger.info("Starting SFT dataset generation process")
    logger.info("Target: 5000 balanced samples from 4 medical datasets")
    logger.info("=" * 60)

    # List of Parquet files to process
    parquet_files = [
        "mediqal_dataset/mediqal_train.parquet",
        "frenchmedmcqa_dataset/frenchmedmcqa_train.parquet",
        "medquad_dataset/medquad_train.parquet",
        "ultramed_dataset/ultramed_train.parquet",
    ]
    logger.info(f"Processing {len(parquet_files)} datasets: {', '.join(parquet_files)}")

    sft_dataset = pd.DataFrame(columns=["question", "answer"])
    total_samples_collected = 0
    target_samples = 5000

    # Process each Parquet file and save the sampled data
    for idx, parquet_file in enumerate(parquet_files, 1):
        # Calculate dynamic sample quota for this dataset
        remaining_datasets = len(parquet_files) - idx + 1
        remaining_quota = target_samples - total_samples_collected
        to_sample = remaining_quota / remaining_datasets

        logger.info(f"[{idx}/{len(parquet_files)}] Processing {parquet_file}")
        logger.info(f"  Target samples: {int(to_sample)} | Remaining quota: {remaining_quota}")

        sampled_df, nb_of_sample = extract_1250_samples(parquet_file, sample=int(to_sample))
        total_samples_collected += nb_of_sample

        sft_dataset = pd.concat([sft_dataset, sampled_df], ignore_index=True)
        logger.info(
            f"  ✓ Added {nb_of_sample} samples | Total in dataset: {len(sft_dataset)}/{target_samples}"
        )

    logger.info("=" * 60)
    logger.info(f"All {len(parquet_files)} datasets processed")
    logger.info(f"Final dataset size: {len(sft_dataset)} rows (Target: {target_samples})")

    output_path = f'{BASE_URL}sft_dataset/sft_dataset.parquet'
    logger.info(f"Saving SFT dataset to {output_path}")
    sft_dataset.to_parquet(output_path, index=False)
    logger.info(
        f"✓ Successfully saved {len(sft_dataset)} samples to {output_path}"
    )
    logger.info("=" * 60)
