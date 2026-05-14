from .utils_cleaning import (
    add_metadata,
    drop_columns,
    merge_raw_data_splits,
    save_cleaned_data_local,
    drop_duplicates,
    transform_correct_answers_to_text,
    create_ground_truth_answer_column,
)
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


    # Step 2: Drop unnecessary columns
    logger.debug("Step 2: Dropping unnecessary columns")
    df = drop_columns(
        df, ["id", "task",  "question_type"]
    )

    # Step 3: Map correct answer indices to their corresponding answer text
    logger.debug("Step 3: Transforming correct answers to text")
    df = transform_correct_answers_to_text(df, MATCH_ANSWER_DICT)
    df = create_ground_truth_answer_column(df)

    # Step 4 : has_clinical_case
    logger.debug("Step 4: Adding has_clinical_case column")
    df["has_clinical_case"] = df["clinical_case"].notna()

    # Step 5: Merge clinical case and question columns
    logger.debug("Step 5: Merging clinical case and question columns")
    df = merge_clinical_case_and_question(df)

    # Step 6: Create a new DataFrame with only 'question' and 'answer' columns
    logger.debug("Step 6: Selecting final columns")
    df_cleaned = df.loc[:,["question", "answer", "medical_subject", "has_clinical_case"]]

    # Step 7: Drop duplicates in the cleaned DataFrame
    logger.debug("Step 7: Dropping duplicates in final dataset")
    df_cleaned = drop_duplicates(df_cleaned)

    # Step 8: Erase non-essential text from questions
    logger.debug("Step 8: Erasing non-essential text from questions")
    df_cleaned = erase_non_essential_text_(df_cleaned)

    # Step 9: Drop questions that ask for false answers
    logger.debug("Step 9: Dropping questions that ask for false answers")
    df_cleaned = drop_question_with_false_answers_asked(df_cleaned)

    # Step 10: Drop questions with numbered proposals in the question body
    logger.debug("Step 10: Dropping questions with numbered proposals")
    df_cleaned = drop_questions_with_numbered_proposals(df_cleaned)

    # Step 11: lower case text
    logger.debug("Step 11: Lowercasing text")
    df_cleaned["question"] = df_cleaned["question"].str.lower()
    df_cleaned["answer"] = df_cleaned["answer"].str.lower()

    # Step 12: Add dataset name column
    logger.debug("Step 12: Adding metadata")
    df_cleaned = add_metadata(
        df_cleaned,
        language="fr",
        question_type="mcq_single",
        confidence_level="medium",
        dataset_name="mediqal"
    )

    logger.info(f"MedIQAL cleaning completed. Output shape: {df_cleaned.shape}")
    return df_cleaned


# === Helper functions ===

def merge_clinical_case_and_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the 'clinical_case' and 'question' columns into a single 'question' column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the MedIQAL dataset.

    Returns:
    pd.DataFrame: A DataFrame with the merged 'question' column.
    """
    logger.debug("Merging 'clinical_case' and 'question' columns")
    mask = df["clinical_case"].notna()
    df.loc[mask, "question"] = df.loc[mask, "clinical_case"] + " " + df.loc[mask, "question"]
    return df

def erase_non_essential_text_(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erase non-essential text from the 'question' column, such as instructions or formatting.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the MedIQAL dataset.

    Returns:
    pd.DataFrame: A DataFrame with cleaned 'question' text.
    """
    logger.debug("Erasing non-essential text from 'question' column")

    df["question"] = df["question"].str.replace("(cochez la réponse juste)", "")
    return df

def drop_questions_with_numbered_proposals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where the question contains numbered proposals (e.g. "1. ", "1) ", "1- ")
    AND the answer is a combination of numbers (e.g. "1+2+3" or "1 3 5").
    These patterns are unsuitable for training as the model would need to output
    a list of indices rather than a proper textual answer.
    """
    logger.debug("Dropping questions with numbered proposals and indexed answers")
    question_has_proposals = df["question"].str.contains(
        r"\n\s*\d+[.\-\)]\s+\S", regex=True
    )
    answer_is_indices = df["answer"].str.contains(
        r"^\s*\d+(\s*[+\s]\s*\d+)+\s*$", regex=True
    )
    mask = question_has_proposals & answer_is_indices
    n_dropped = mask.sum()
    logger.debug(f"Dropping {n_dropped} rows with numbered proposals")
    return df[~mask]


def drop_question_with_false_answers_asked(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where the question asks for false answers, as they may not be suitable for training.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the MedIQAL dataset.

    Returns:
    pd.DataFrame: A DataFrame with rows containing false answer questions removed.
    """
    logger.debug("Dropping questions that ask for false answers")
    mask = df["question"].str.contains("cochez la réponse fausse", case=False)
    df = df[~mask]
    return df
if __name__ == "__main__":
    from datasets import load_from_disk
    from config.paths import PROCESSED_DATA_DIR, RAW_DATA_GCS_URL

    datasets = load_from_disk(f"{RAW_DATA_GCS_URL}/mediqal_datasets/mcqu_medical/")
    
    df = merge_raw_data_splits(datasets)
    df_cleaned = clean_mediqal(df)
    save_cleaned_data_local(
        df_cleaned,
        PROCESSED_DATA_DIR
        / "mediqal_dataset"
        / "mediqal.parquet",
    )
