# ============================================================
# preprocessing.py
# Data preprocessing
# ============================================================

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import TEST_SIZE, RANDOM_STATE


def encode_labels(y):
    """
    Convert string labels into numerical labels.
    """

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y
    )

    y_categorical = tf.keras.utils.to_categorical(
        y_encoded
    )

    return (
        y_encoded,
        y_categorical,
        label_encoder
    )


def add_channel_dimension(X):
    """
    Add CNN channel dimension.

    (samples, 40, 174)
    ->
    (samples, 40, 174, 1)
    """

    return X[..., np.newaxis]


def split_dataset(
    X,
    y_categorical,
    y_encoded
):
    """
    Perform stratified train/test split.
    """

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y_categorical,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_encoded
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def normalize_features(
    X_train,
    X_test
):
    """
    Normalize test data using statistics
    calculated from training data only.
    """

    mean = np.mean(
        X_train
    )

    std = np.std(
        X_train
    )

    X_train_normalized = (
        X_train - mean
    ) / (
        std + 1e-8
    )

    X_test_normalized = (
        X_test - mean
    ) / (
        std + 1e-8
    )

    return (
        X_train_normalized,
        X_test_normalized,
        mean,
        std
    )
