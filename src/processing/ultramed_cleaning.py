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


def clean_ultramed_for_SFT(df: pd.DataFrame) -> pd.DataFrame:
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

    # Step 3: lower case text
    logger.debug("Step 3: Lowercasing text")
    df["question"] = df["question"].str.lower()
    df["answer"] = df["answer"].str.lower()

    # Step 4: Add dataset name column
    logger.debug("Step 4: Adding dataset name column")
    df["dataset_name"] = "ultramed"

    return df

def clean_ultramed_for_DPO(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the UltraMedical dataset for DPO training by performing the following steps:
    1. Transform the conversation format to a question-chosen-rejected format.
    2. Drop duplicates.
    3. Lowercase text in question and answer columns.
    4. Add dataset name column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the UltraMedical dataset.

    Returns:
    pd.DataFrame: A cleaned DataFrame with 'question', 'chosen', 'rejected', and 'dataset_name' columns.
    """
    logger.info("Starting UltraMedical cleaning process")
    logger.debug(f"Input DataFrame shape: {df.shape}")

    # Step 1: transform conversation format to question-chosen-rejected format for DPO training
    logger.debug("Step 1: Transforming conversation format to question-chosen-rejected format for DPO training")
    df = transform_conversation_dpo_dataset(df)

    # Step 2: Drop duplicates
    logger.debug("Step 2: Dropping duplicates")
    df = drop_duplicates(df)
    logger.info(f"UltraMedical cleaning completed. Output shape: {df.shape}")

    # Step 3: lower case text
    logger.debug("Step 3: Lowercasing text")
    df["question"] = df["question"].str.lower()
    df["chosen"] = df["chosen"].str.lower()
    df["rejected"] = df["rejected"].str.lower()

    # Step 4: Add dataset name column
    logger.debug("Step 4: Adding dataset name column")
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

    qa_df = df.apply(extract_qa, axis=1)

    logger.info(f"Transformation completed. Output shape: {qa_df.shape}")
    return qa_df

def extract_dpo(row):
    chosen_col = row.get("chosen", [])
    rejected_col = row.get("rejected", [])
    question = next((m["content"] for m in chosen_col if m["role"] == "user"), None)
    chosen = next((m["content"] for m in chosen_col if m["role"] == "assistant"), None)
    rejected = next((m["content"] for m in rejected_col if m["role"] == "assistant"), None)
    return pd.Series({"question": question, "chosen": chosen, "rejected": rejected})

def transform_conversation_dpo_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the UltraMedical dataset from a conversation format to a question-chosen-rejected format for DPO training.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the UltraMedical dataset.

    Returns:
    pd.DataFrame: A cleaned DataFrame with 'question', 'chosen', and 're
    """
    logger.info("Transforming UltraMedical dataset to question-chosen-rejected format")

    dpo_df = df.apply(extract_dpo, axis=1)

    logger.info(f"Transformation completed. Output shape: {dpo_df.shape}")
    return dpo_df

if __name__ == "__main__":
    from datasets import load_from_disk
    from config.paths import PROCESSED_DATA_DIR, RAW_DATA_GCS_URL

    datasets = load_from_disk(f"{RAW_DATA_GCS_URL}/UltraMedical_dataset")

    df = merge_raw_data_splits(datasets)
    df_SFT = clean_ultramed_for_SFT(df)
    df_DPO = clean_ultramed_for_DPO(df)
    save_cleaned_data_local(
        df_SFT,
        PROCESSED_DATA_DIR
        / "ultramed_dataset"
        / f"ultramed_sft.parquet",
    )
    save_cleaned_data_local(
        df_DPO,
        PROCESSED_DATA_DIR
        / "ultramed_dataset"
        / f"ultramed_dpo.parquet",
    )
