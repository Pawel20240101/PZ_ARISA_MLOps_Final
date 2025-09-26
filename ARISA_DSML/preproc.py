"""Functions for preprocessing the data."""

import os
import shutil
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
from sklearn.model_selection import train_test_split

from ARISA_DSML.config import (
    DATASET,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    categorical,
    target,
)


def get_raw_data(dataset: str = DATASET) -> None:
    """Download dataset from Kaggle."""
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)
    download_folder.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(dataset, path=str(download_folder), unzip=True)

    # Przeniesienie pliku z folderu 2020
    base_dir = download_folder
    file_to_move = base_dir / "2020" / "heart_2020_cleaned.csv"
    target_location = base_dir / "heart_2020_cleaned.csv"

    if file_to_move.exists():
        shutil.move(str(file_to_move), str(target_location))
        logger.info(f"Moved {file_to_move} to {target_location}")

    # Usunięcie zbędnych folderów
    for folder in ["2020", "2022"]:
        folder_path = base_dir / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            logger.info(f"Removed folder {folder_path}")


def preprocess_df(file: str | Path) -> tuple[Path, Path]:
    """Preprocess dataset."""
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)

    logger.info(f"Original columns: {df_data.columns.tolist()}")
    logger.info(f"Target column '{target}' unique values: {df_data[target].unique()}")

    # Konwersja Yes/No na 1/0 dla zmiennej target
    df_data[target] = (df_data[target] == "Yes").astype(int)

    # Walidacja kolumn kategorycznych
    missing_categorical = [col for col in categorical if col not in df_data.columns]
    if missing_categorical:
        logger.warning(f"Missing categorical columns: {missing_categorical}")

    # Podział train/test
    df_train, df_test = train_test_split(
        df_data, test_size=0.2, random_state=42, stratify=df_data[target]
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    logger.info(f"Train target distribution: {df_train[target].value_counts().to_dict()}")

    return train_path, test_path


if __name__ == "__main__":
    logger.info("Getting datasets")
    get_raw_data()

    logger.info("Preprocessing heart_2020_cleaned.csv")
    train_path, test_path = preprocess_df(RAW_DATA_DIR / "heart_2020_cleaned.csv")
    logger.info(f"Saved to {train_path} and {test_path}")
