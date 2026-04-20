from functools import lru_cache
from pathlib import Path

import pandas as pd
import pandas
from config.logger import logger


# === Cleaning helper functions ===
def save_cleaned_data_local(df: pd.DataFrame, destination_path: str | Path) -> None:
    """
    Save the cleaned DataFrame as a Parquet file on the local filesystem.

    The parent directory is created if it does not exist. DVC is expected to
    track the output and sync it to the configured remote.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame to be saved.
    destination_path (str | Path): Local path of the output Parquet file.
    """
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving cleaned data locally to: {destination_path}")
    logger.debug(f"DataFrame shape: {df.shape}")
    df.to_parquet(destination_path, index=False)
    size_mb = destination_path.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully wrote {size_mb:.2f} MB to {destination_path}")


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_drop (list): List of column names to drop.

    Returns:
    pd.DataFrame: DataFrame with specified columns dropped.
    """
    logger.info(f"Dropping columns: {columns_to_drop}")
    logger.debug(f"DataFrame shape before: {df.shape}")
    result = df.drop(columns=columns_to_drop)
    logger.debug(f"DataFrame shape after: {result.shape}")
    logger.info(f"Successfully dropped {len(columns_to_drop)} column(s)")
    return result


def save_cleaned_data_to_gcs(
    df: pd.DataFrame, bucket_name: str, destination_blob_name: str
):
    """
    Save the cleaned DataFrame to a Google Cloud Storage bucket.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame to be saved.
    bucket_name (str): The name of the GCS bucket where the file will be saved.
    destination_blob_name (str): The name of the destination blob in the GCS bucket.

    Returns:
    None
    """
    from google.cloud import storage
    import io

    logger.info(
        f"Saving cleaned data to GCS bucket: {bucket_name}/{destination_blob_name}"
    )
    logger.debug(f"DataFrame shape: {df.shape}")

    try:
        # Create a GCS client and get the bucket
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Convert DataFrame to Parquet and upload to GCS
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        file_size_mb = len(parquet_buffer.getvalue()) / (1024 * 1024)
        logger.debug(f"Parquet file size: {file_size_mb:.2f} MB")
        blob.upload_from_string(
            parquet_buffer.getvalue(), content_type="application/octet-stream"
        )
        logger.info(f"Successfully uploaded cleaned data to GCS")
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}", exc_info=True)
        raise


def drop_duplicates(df, subset=None):
    """
    Drop duplicate rows from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    subset (list, optional): List of column names to consider for identifying duplicates.
                             If None, considers all columns.

    Returns:
    pd.DataFrame: DataFrame with duplicate rows dropped.
    """
    logger.info(f"Removing duplicate rows (subset={subset})")
    logger.debug(f"DataFrame shape before: {df.shape}")

    num_duplicates = df.duplicated(subset=subset).sum()
    logger.info(f"Found {num_duplicates} duplicate row(s)")

    result = df.drop_duplicates(subset=subset)
    logger.debug(f"DataFrame shape after: {result.shape}")
    logger.info(f"Successfully removed {len(df) - len(result)} duplicate row(s)")

    return result


def transform_correct_answers_to_text(
    df: pd.DataFrame, match_answer_dict: dict
) -> pd.DataFrame:
    """
    Transform the 'correct_answers' column from indices to text.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'correct_answers' column.
    match_answer_dict (dict): A dictionary mapping answer indices to their corresponding text.

    Returns:
    pd.DataFrame: The DataFrame with a new 'correct_answer_text' column.
    """
    logger.debug("Mapping correct answer indices to text")
    df["correct_answer_text"] = df["correct_answers"].map(match_answer_dict)

    # Check for unmapped values
    unmapped_count = df["correct_answer_text"].isna().sum()
    if unmapped_count > 0:
        logger.warning(f"Found {unmapped_count} unmapped answer index values")

    logger.debug(f"Successfully mapped {len(df) - unmapped_count} answer indices")
    return df


def create_ground_truth_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'answer' that contains the text of the correct answer.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'correct_answer_text' column.

    Returns:
    pd.DataFrame: The DataFrame with a new 'answer' column.
    """
    logger.debug("Creating ground truth answer column")
    try:
        df["answer"] = df.apply(lambda row: row[row["correct_answer_text"]], axis=1)
        logger.debug(f"Successfully created 'answer' column for {len(df)} rows")
    except Exception as e:
        logger.error(
            f"Error creating ground truth answer column: {str(e)}", exc_info=True
        )
        raise
    return df


def merge_raw_data_splits(datasets) -> pd.DataFrame:
    """
    Merge multiple raw data splits (e.g., train, validation, test) into a single DataFrame.

    Parameters:
    dataframes (dict): A dictionary where keys are split names and values are the corresponding DataFrames.

    Returns:
    pd.DataFrame: A single merged DataFrame containing all splits.
    """
    logger.info(f"Merging raw data splits")

    merged_df = pd.DataFrame()

    for dataset in datasets.values():
        df = dataset.to_pandas()
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    logger.info(
        f"Successfully merged splits into a single DataFrame with shape: {merged_df.shape}"
    )
    return merged_df

# === Metadata helper functions ===
def add_metadata(df:pandas.DataFrame, language:str, question_type:str, confidence_level:str, dataset_name:str) -> pandas.DataFrame:
    """
    Add metadata columns to a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    language (str): The language of the dataset (e.g., 'en', 'fr').
    question_type (str): The type of questions in the dataset (e.g., 'mcq_single', 'mcq_single', 'open_qa','conversational' ).
    confidence_level (str): The confidence level of the dataset (e.g., 'low', 'medium', 'high').
    dataset_name (str): The name of the dataset.

    Returns:
    pd.DataFrame: The DataFrame with metadata columns added.
    """
    df["language"] = language
    df["question_type"] = question_type
    df["confidence_level"] = confidence_level
    df["dataset_name"] = dataset_name
    return df

from functools import lru_cache
@lru_cache(maxsize=1)
def _get_qwen_tokenizer():
    """Chargé une seule fois par process (lru_cache)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")


def add_token_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Ajoute une colonne token_count_<col> pour chaque colonne texte spécifiée.

    Args:
        df: DataFrame contenant les colonnes texte à tokenizer.
        columns: Liste des noms de colonnes (ex: ["question", "answer"]).

    Returns:
        DataFrame enrichi avec les colonnes de comptage.
    """
    logger.info(f"Computing token counts with Qwen3 tokenizer for columns: {columns}")
    tokenizer = _get_qwen_tokenizer()

    for col in columns:
        # batch=True via une list comprehension : plus rapide qu'un .apply
        texts = df[col].fillna("").tolist()
        encodings = tokenizer(texts, add_special_tokens=False)
        df[f"token_count_{col}"] = [len(ids) for ids in encodings["input_ids"]]
        logger.info(
            f"  {col}: min={df[f'token_count_{col}'].min()}, "
            f"median={df[f'token_count_{col}'].median():.0f}, "
            f"max={df[f'token_count_{col}'].max()}"
        )

    return df

# === Sampling helper functions ===

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


def collect_balanced_samples(
    parquet_files: list[str],
    base_dir: Path,
    target_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Collect a balanced sample across multiple Parquet files.

    Distributes target_samples evenly across all files. If a file has fewer
    rows than its share, the shortfall is redistributed to the remaining files.

    Args:
        parquet_files: List of Parquet filenames relative to base_dir.
        base_dir: Directory containing the Parquet files.
        target_samples: Total number of rows to collect.
        random_state: Seed used by pandas.DataFrame.sample for reproducibility.

    Returns:
        A single concatenated DataFrame with at most target_samples rows.
    """
    collected = []
    total_samples_collected = 0

    for idx, parquet_file in enumerate(parquet_files, 1):
        remaining_datasets = len(parquet_files) - idx + 1
        remaining_quota = target_samples - total_samples_collected
        to_sample = remaining_quota // remaining_datasets

        logger.info(f"[{idx}/{len(parquet_files)}] Processing {parquet_file}")
        logger.info(f"  Target samples: {to_sample} | Remaining quota: {remaining_quota}")

        sampled_df, nb_of_sample = extract_samples(
            base_dir / parquet_file,
            sample=to_sample,
            random_state=random_state,
        )
        total_samples_collected += nb_of_sample
        collected.append(sampled_df)
        logger.info(
            f"  Added {nb_of_sample} samples | Total in dataset: {total_samples_collected}/{target_samples}"
        )

    return pd.concat(collected, ignore_index=True)

# === Splitting helper functions ===