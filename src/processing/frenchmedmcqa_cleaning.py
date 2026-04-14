from .utils_cleaning import drop_columns, save_cleaned_data_to_gcs, drop_duplicates, transform_correct_answers_to_text, create_ground_truth_answer_column
import pandas as pd
from config.logger import logger

MATCH_ANSWER_DICT = {
    0: "answer_a",
    1: "answer_b",
    2: "answer_c",
    3: "answer_d",
    4: "answer_e",
}

# === Main cleaning function ===

def clean_frenchmedmcqa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the FrenchMedMCQA dataset by performing the following steps:
    1. Drop unnecessary columns.
    2. Map the correct answer indices to their corresponding answer text.
    3. Create a new DataFrame with only the 'question' and 'answer' columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the FrenchMedMCQA dataset.

    Returns:
    pd.DataFrame: A cleaned DataFrame with 'question' and 'answer' columns.
    """
    logger.info("Starting FrenchMedMCQA cleaning process")
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug("Step 1: Dropping duplicates")
    df = drop_duplicates(df)
    # Step 2: Drop unnecessary columns
    logger.debug("Step 2: Dropping unnecessary columns")
    df = drop_columns(df, ["id", "number_correct_answers"])

    # Step 3: Map correct answer indices to their corresponding answer text
    logger.debug("Step 3: Transforming correct answers to text")
    df = transform_correct_answers_to_text(df, MATCH_ANSWER_DICT)
    df = create_ground_truth_answer_column(df)

    # Step 4: Create a new DataFrame with only 'question' and 'answer' columns
    logger.debug("Step 4: Selecting final columns")
    df_cleaned = df.loc[:, ["question", "answer"]]

    # Step 5: Drop duplicates in the cleaned DataFrame
    logger.debug("Step 5: Dropping duplicates")
    df_cleaned = drop_duplicates(df_cleaned)

    logger.info(f"FrenchMedMCQA cleaning completed. Output shape: {df_cleaned.shape}")
    return df_cleaned

# === Helper functions ===


if __name__ == "__main__":
    import pandas as pd
    from datasets import load_from_disk
    # Load the dataset
    datasets = load_from_disk("gs://p14-medical-data/raw_data/frenchmedmcqa_dataset/")
    
    # Iterate over splits (train, validation, test)
    for split_name, dataset in datasets.items():
        df = dataset.to_pandas()
        df_cleaned = clean_frenchmedmcqa(df)
        save_cleaned_data_to_gcs(df_cleaned, "p14-medical-data", f"processed_data/frenchmedmcqa_dataset/frenchmedmcqa_{split_name}.parquet")
