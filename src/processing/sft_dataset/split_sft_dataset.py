import yaml
import pandas as pd
from config.logger import logger
from config.paths import PROJECT_ROOT, SFT_AUGMENTED_DATASET_PATH, SFT_TRAIN_DATASET_PATH, SFT_VAL_DATASET_PATH, SFT_TEST_DATASET_PATH
from src.processing.utils_cleaning import split_dataset, save_cleaned_data_local

def main():
    # 1. Charger les paramètres
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        params = yaml.safe_load(f)["sft"]

    random_state = params.get("random_state", 42)
    val_size = params.get("val_size", 0.1)
    test_size = params.get("test_size", 0.1)

    logger.info(f"Loading dataset from: {SFT_AUGMENTED_DATASET_PATH}")
    df = pd.read_parquet(SFT_AUGMENTED_DATASET_PATH)

    # 3. Utiliser votre fonction de utils_cleaning.py
    train_df, val_df, test_df = split_dataset(
        df=df,
        random_state=random_state,
        val_size=val_size,
        test_size=test_size
    )

    # 4. Sauvegarder les splits
    save_cleaned_data_local(train_df, SFT_TRAIN_DATASET_PATH)
    save_cleaned_data_local(val_df, SFT_VAL_DATASET_PATH)
    save_cleaned_data_local(test_df, SFT_TEST_DATASET_PATH)

    logger.info("Successfully generated train, val, and test splits.")

if __name__ == "__main__":
    main()