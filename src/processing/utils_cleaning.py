import pandas as pd
from config.logger import logger


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


def save_cleaned_data_to_gcs(df: pd.DataFrame, bucket_name: str, destination_blob_name: str):
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

    logger.info(f"Saving cleaned data to GCS bucket: {bucket_name}/{destination_blob_name}")
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
        blob.upload_from_string(parquet_buffer.getvalue(), content_type='application/octet-stream')
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

def transform_correct_answers_to_text(df: pd.DataFrame, match_answer_dict: dict) -> pd.DataFrame:
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
        logger.error(f"Error creating ground truth answer column: {str(e)}", exc_info=True)
        raise
    return df