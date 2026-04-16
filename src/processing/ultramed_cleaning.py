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


def clean_ultramed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the UltraMedical dataset by performing the following steps:
    1. Transform the conversation format to a question-answer format.
    2. Drop unnecessary columns.
    3. Map the correct answer indices to their corresponding answer text.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the UltraMedical dataset.

    Returns:
    pd.DataFrame: A cleaned DataFrame with 'question' and 'answer' columns.
    """
    logger.info("Starting UltraMedical cleaning process")
    logger.debug(f"Input DataFrame shape: {df.shape}")

    # Step 1: transform conversation format to question-answer format
    logger.debug("Step 1: Transforming conversation format to question-answer format")
    df = transform_conversation_to_qa_format(df)

    # Step 2: Drop duplicates
    logger.debug("Step 2: Dropping duplicates")
    df = drop_duplicates(df)
    logger.info(f"UltraMedical cleaning completed. Output shape: {df.shape}")

    # Step 3: Add dataset name column
    logger.debug("Step 3: Adding dataset name column")
    df["dataset_name"] = "ultramed"

    return df

def extract_qa(row):
    chosen = row.get("chosen", [])
    question = next((m["content"] for m in chosen if m["role"] == "user"), None)
    answer = next((m["content"] for m in chosen if m["role"] == "assistant"), None)
    return pd.Series({"question": question, "answer": answer})

def transform_conversation_to_qa_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the UltraMedical dataset from a conversation format to a question-answer format.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the UltraMedical dataset.

    Returns:
    pd.DataFrame: A cleaned DataFrame with 'question' and 'answer' columns.
    """
    logger.info("Transforming UltraMedical dataset to question-answer format")

    questions = []
    answers = []

    result = df.apply(extract_qa, axis=1)

    # Create a new DataFrame with 'question' and 'answer' columns
    qa_df = pd.DataFrame({"question": questions, "answer": answers})

    logger.info(f"Transformation completed. Output shape: {qa_df.shape}")
    return qa_df


if __name__ == "__main__":
    from datasets import load_from_disk
    from config.paths import PROCESSED_DATA_DIR, RAW_DATA_GCS_URL

    datasets = load_from_disk(f"{RAW_DATA_GCS_URL}/UltraMedical_dataset")

    df = merge_raw_data_splits(datasets)
    df_cleaned = clean_ultramed(df)
    save_cleaned_data_local(
        df_cleaned,
        PROCESSED_DATA_DIR
        / "ultramed_dataset"
        / f"ultramed.parquet",
    )
