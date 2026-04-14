import pandas as pd


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_drop (list): List of column names to drop.

    Returns:
    pd.DataFrame: DataFrame with specified columns dropped.
    """
    return df.drop(columns=columns_to_drop)


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

    # Create a GCS client and get the bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Convert DataFrame to Parquet and upload to GCS
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    blob.upload_from_string(parquet_buffer.getvalue(), content_type='application/octet-stream')

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
    return df.drop_duplicates(subset=subset)