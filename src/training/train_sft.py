from datasets import Dataset
import pandas as pd
from pathlib import Path

from config.logger import logger
from config.paths import PARAMS_PATH, PROJECT_ROOT,SFT_TRAIN_DATASET_PATH,SFT_VAL_DATASET_PATH
from src.training.utils_training import load_dataset, tokenize_chat


def transform_ds_from_pandas_to_hf(dataset: pd.DataFrame) -> Dataset:
    logger.info("Transforming dataset from pandas DataFrame to Hugging Face Dataset...")
    hf_dataset = Dataset.from_pandas(dataset)
    logger.info("Dataset transformed successfully to Hugging Face Dataset")
    return hf_dataset

def apply_tokenisation(dataset_hf: Dataset) -> Dataset:
    logger.info("Applying tokenization to the dataset...")
    tokenized_dataset = dataset_hf.map(
        lambda x: tokenize_chat(x["question"], x["answer"]),
        batched=False,
        remove_columns=dataset_hf.column_names,
    )
    logger.info("Tokenization applied successfully to the dataset")
    return tokenized_dataset

def tokenize_flow(pd_dataset_path: Path) -> Dataset:
    ds_name = pd_dataset_path.stem
    logger.info(f"===== Tokenizing {ds_name} =====")

    pd_dataset = load_dataset(pd_dataset_path)
    hf_dataset = transform_ds_from_pandas_to_hf(pd_dataset)
    tokenized_dataset = apply_tokenisation(hf_dataset)
    
    logger.info(f"===== {ds_name} tokenized successfully =====")
    return tokenized_dataset
