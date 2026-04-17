from .utils_cleaning import (
    drop_columns,
    merge_raw_data_splits,
    save_cleaned_data_local,
    drop_duplicates,
    transform_correct_answers_to_text,
    create_ground_truth_answer_column,
)
import pandas as pd
from config.logger import logger

# === Main cleaning function ===


def clean_medquad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the MedQuad dataset by performing the following steps:
    1. Drop unnecessary columns.
    2. Map the correct answer indices to their corresponding answer text.
    3. Create a new DataFrame with only the 'question' and 'answer' columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the MedQuad dataset.

    Returns:
    pd.DataFrame: A cleaned DataFrame with 'question' and 'answer' columns.
    """
    logger.info("Starting MedQuad cleaning process")
    logger.debug(f"Input DataFrame shape: {df.shape}")

    # Step 1: Drop duplicates
    logger.debug("Step 1: Dropping duplicates")
    df = drop_duplicates(df)

    # Step 2: Drop unnecessary columns
    logger.debug("Step 2: Dropping unnecessary columns")
    df = drop_columns(df, ["qtype"])

    # Step 3: Rename columns to match the expected format
    logger.debug("Step 3: Renaming columns")
    df.rename(columns={"Question": "question", "Answer": "answer"}, inplace=True)

    # Step 4: lower case text
    logger.debug("Step 4: Lowercasing text")
    df["question"] = df["question"].str.lower()
    df["answer"] = df["answer"].str.lower()

    # Step 5: Add dataset name column
    logger.debug("Step 5: Adding dataset name column")
    df["dataset_name"] = "medquad"

    logger.info(f"MedQuad cleaning completed. Output shape: {df.shape}")
    return df


if __name__ == "__main__":
    from datasets import load_from_disk
    from config.paths import PROCESSED_DATA_DIR, RAW_DATA_GCS_URL

    datasets = load_from_disk(f"{RAW_DATA_GCS_URL}/MedQuad_dataset")
    
    df = merge_raw_data_splits(datasets)
    df_cleaned = clean_medquad(df)
    save_cleaned_data_local(
        df_cleaned,
        PROCESSED_DATA_DIR
        / "medquad_dataset"
        / f"medquad.parquet",
    )
