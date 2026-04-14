from .utils_cleaning import (
    drop_columns,
    save_cleaned_data_to_gcs,
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

    return df


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

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        conversation = row[
            "chosen"
        ]  # Assuming 'chosen' column contains the conversation
        if (
            len(conversation) >= 2
        ):  # Ensure there are at least two messages (question and answer)
            user_msgs = [
                msg["content"] for msg in conversation if msg["role"] == "user"
            ]
            assistant_msgs = [
                msg["content"] for msg in conversation if msg["role"] == "assistant"
            ]
            if user_msgs and assistant_msgs:
                questions.append(user_msgs[0])
                answers.append(assistant_msgs[0])

    # Create a new DataFrame with 'question' and 'answer' columns
    qa_df = pd.DataFrame({"question": questions, "answer": answers})

    logger.info(f"Transformation completed. Output shape: {qa_df.shape}")
    return qa_df


if __name__ == "__main__":
    import pandas as pd
    from datasets import load_from_disk

    datasets = load_from_disk("gs://p14-medical-data/raw_data/UltraMedical_dataset")
    
    # Iterate over splits (train, validation, test)
    for split_name, dataset in datasets.items():
        df = dataset.to_pandas()
        df_cleaned = clean_ultramed(df)
        save_cleaned_data_to_gcs(df_cleaned, "p14-medical-data", f"processed_data/ultramed_dataset/ultramed_{split_name}.parquet")