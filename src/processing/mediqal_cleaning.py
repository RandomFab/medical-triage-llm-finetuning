from .utils_cleaning import drop_columns, save_cleaned_data_to_gcs, drop_duplicates, transform_correct_answers_to_text, create_ground_truth_answer_column
import pandas as pd
from config.logger import logger

MATCH_ANSWER_DICT = {
    "A": "answer_a",
    "B": "answer_b",
    "C": "answer_c",
    "D": "answer_d",
    "E": "answer_e",
}

# === Main cleaning function ===

def clean_mediqal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the MedIQAL dataset by performing the following steps:
    1. Drop unnecessary columns.
    2. Map the correct answer indices to their corresponding answer text.
    3. Create a new DataFrame with only the 'question' and 'answer' columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the MedIQAL dataset.

    Returns:
    pd.DataFrame: A cleaned DataFrame with 'question' and 'answer' columns.
    """
    logger.info("Starting MedIQAL cleaning process")
    logger.debug(f"Input DataFrame shape: {df.shape}")

    # Step 1: Drop duplicates
    logger.debug("Step 1: Dropping duplicates")
    df = drop_duplicates(df)

    # Step 2: Drop clinical cases lines
    logger.debug("Step 2: Dropping clinical cases lines")
    df = drop_clinical_cases(df)

    # Step 3: Drop unnecessary columns
    logger.debug("Step 3: Dropping unnecessary columns")
    df = drop_columns(df, ["id", "task", "clinical_case","medical_subject","question_type"])

    # Step 4: Map correct answer indices to their corresponding answer text
    logger.debug("Step 4: Transforming correct answers to text")
    df = transform_correct_answers_to_text(df, MATCH_ANSWER_DICT)
    df = create_ground_truth_answer_column(df)

    # Step 5: Create a new DataFrame with only 'question' and 'answer' columns
    logger.debug("Step 5: Selecting final columns")
    df_cleaned = df.loc[:, ["question", "answer"]]

    # Step 6: Drop duplicates in the cleaned DataFrame
    logger.debug("Step 6: Dropping duplicates in final dataset")
    initial_clean_shape = df_cleaned.shape[0]
    df_cleaned = drop_duplicates(df_cleaned)

    logger.info(f"MedIQAL cleaning completed. Output shape: {df_cleaned.shape}")
    return df_cleaned

# === Helper functions ===
def drop_clinical_cases(df):
    """Keep only rows where clinical_case column is not null."""
    logger.debug(f"drop_clinical_cases: Input shape {df.shape}")
    df_filtered = df[df["clinical_case"].notna()]
    logger.debug(f"drop_clinical_cases: Output shape {df_filtered.shape} ({len(df) - len(df_filtered)} rows removed)")
    return df_filtered

if __name__ == "__main__":
    import pandas as pd
    from datasets import load_from_disk
    # Load the dataset
    datasets = load_from_disk("gs://p14-medical-data/raw_data/mediqal_datasets/mcqu_medical/")
    
    # Iterate over splits (train, validation, test)
    for split_name, dataset in datasets.items():
        df = dataset.to_pandas()
        df_cleaned = clean_mediqal(df)
        save_cleaned_data_to_gcs(df_cleaned, "p14-medical-data", f"processed_data/mediqal_dataset/mediqal_{split_name}.parquet")
