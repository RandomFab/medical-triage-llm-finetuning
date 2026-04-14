from json import load

from .utils_cleaning import drop_columns, save_cleaned_data_to_gcs, drop_duplicates, transform_correct_answers_to_text, create_ground_truth_answer_column
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
    df = drop_columns(df, ['qtype'])

    # Step 3: Rename columns to match the expected format
    logger.debug("Step 3: Renaming columns")
    df.rename(columns={
        'Question': 'question',
        'Answer': 'answer'
    }, inplace=True)


    logger.info(f"MedQuad cleaning completed. Output shape: {df.shape}")
    return df

if __name__ == "__main__":
    from datasets import load_from_disk

    datasets = load_from_disk("gs://p14-medical-data/raw_data/MedQuad_dataset")

     # Iterate over splits (train, validation, test)
    for split_name, dataset in datasets.items():
        df = dataset.to_pandas()
        df_cleaned = clean_medquad(df)
        save_cleaned_data_to_gcs(df_cleaned, "p14-medical-data", f"processed_data/medquad_dataset/medquad_{split_name}.parquet")
