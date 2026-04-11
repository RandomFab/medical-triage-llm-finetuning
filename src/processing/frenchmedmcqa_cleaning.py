from .utils_cleaning import drop_columns, save_cleaned_data_to_gcs
import pandas as pd
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
    # Step 1: Drop unnecessary columns
    df = drop_columns(df, ["id", "number_correct_answers"])

    # Step 2: Map correct answer indices to their corresponding answer text
    df = transform_correct_answers_to_text(df)
    df = create_ground_truth_answer_column(df)

    # Step 3: Create a new DataFrame with only 'question' and 'answer' columns
    df_cleaned = df.loc[:, ["question", "answer"]]

    return df_cleaned

# === Helper functions ===
def transform_correct_answers_to_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the 'correct_answers' column from indices to text.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'correct_answers' column.

    Returns:
    pd.DataFrame: The DataFrame with a new 'correct_answer_text' column.
    """
    df["correct_answer_text"] = df["correct_answers"].map(MATCH_ANSWER_DICT)
    return df


def create_ground_truth_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'answer' that contains the text of the correct answer.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'correct_answer_text' column.

    Returns:
    pd.DataFrame: The DataFrame with a new 'answer' column.
    """
    df["answer"] = df.apply(lambda row: row[row["correct_answer_text"]], axis=1)
    return df

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
