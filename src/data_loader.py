
# ============================================================
# data_loader.py
# Dataset loading and preparation
# ============================================================

import os
import pandas as pd

from config import DATASET_PATH, TARGET_CLASSES


def load_metadata():
    """
    Load UrbanSound8K metadata.
    """

    metadata_path = os.path.join(
        DATASET_PATH,
        "metadata",
        "UrbanSound8K.csv"
    )

    metadata = pd.read_csv(
        metadata_path
    )

    return metadata


def filter_target_classes(metadata):
    """
    Select only the five target sound classes.
    """

    filtered_metadata = metadata[
        metadata["class"].isin(TARGET_CLASSES)
    ].copy()

    return filtered_metadata


def create_audio_paths(metadata):
    """
    Create complete paths to WAV files.
    """

    metadata["path"] = metadata.apply(
        lambda row: os.path.join(
            DATASET_PATH,
            f"fold{row['fold']}",
            row["slice_file_name"]
        ),
        axis=1
    )

    return metadata


def verify_audio_files(metadata):
    """
    Remove files that do not exist.
    """

    metadata["file_exists"] = (
        metadata["path"].apply(os.path.exists)
    )

    metadata = metadata[
        metadata["file_exists"]
    ].copy()

    return metadata


def load_dataset():
    """
    Complete dataset loading pipeline.
    """

    metadata = load_metadata()

    metadata = filter_target_classes(
        metadata
    )

    metadata = create_audio_paths(
        metadata
    )

    metadata = verify_audio_files(
        metadata
    )

    metadata.reset_index(
        drop=True,
        inplace=True
    )

    return metadata


if __name__ == "__main__":

    metadata = load_dataset()

    print(
        "Number of samples:",
        len(metadata)
    )

    print(
        "\nClasses:"
    )

    print(
        metadata["class"].unique()
    )
