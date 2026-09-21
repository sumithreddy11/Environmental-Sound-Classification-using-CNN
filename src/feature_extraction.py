
# ============================================================
# feature_extraction.py
# Audio processing and MFCC extraction
# ============================================================

import numpy as np
import librosa

from config import N_MFCC, MAX_LEN


def extract_mfcc(file_path):
    """
    Load an audio file and extract a fixed-size MFCC
    representation.

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    numpy.ndarray
        MFCC feature matrix of shape
        (N_MFCC, MAX_LEN)
    """

    try:

        audio, sr = librosa.load(
            file_path,
            sr=None
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        if mfcc.shape[1] < MAX_LEN:

            padding = MAX_LEN - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                (
                    (0, 0),
                    (0, padding)
                ),
                mode="constant"
            )

        # ----------------------------------------------------
        # Truncation
        # ----------------------------------------------------

        else:

            mfcc = mfcc[:, :MAX_LEN]

        return mfcc

    except Exception as error:

        print(
            f"Error processing {file_path}: {error}"
        )

        return None


def extract_dataset_features(metadata):
    """
    Extract MFCC features from all audio files.
    """

    features = []
    labels = []

    total = len(metadata)

    for index, row in metadata.iterrows():

        mfcc = extract_mfcc(
            row["path"]
        )

        if mfcc is not None:

            features.append(mfcc)
            labels.append(row["class"])

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1}/{total} files"
            )

    X = np.array(features)

    y = np.array(labels)

    return X, y
