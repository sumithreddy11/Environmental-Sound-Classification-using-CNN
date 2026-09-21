
# ============================================================
# train.py
# Complete CNN training pipeline
# ============================================================

import numpy as np

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from data_loader import load_dataset

from feature_extraction import (
    extract_dataset_features
)

from preprocessing import (
    encode_labels,
    add_channel_dimension,
    split_dataset,
    normalize_features
)

from model import build_model

from config import (
    BEST_MODEL_PATH,
    EPOCHS,
    BATCH_SIZE
)


def main():

    # ========================================================
    # 1. Load dataset
    # ========================================================

    print("\nLoading dataset...")

    metadata = load_dataset()

    print(
        "Number of samples:",
        len(metadata)
    )

    # ========================================================
    # 2. Extract MFCC features
    # ========================================================

    print("\nExtracting MFCC features...")

    X, y = extract_dataset_features(
        metadata
    )

    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Label shape:",
        y.shape
    )

    # ========================================================
    # 3. Encode labels
    # ========================================================

    (
        y_encoded,
        y_categorical,
        label_encoder
    ) = encode_labels(y)

    print(
        "\nClasses:",
        label_encoder.classes_
    )

    # ========================================================
    # 4. Add CNN channel
    # ========================================================

    X = add_channel_dimension(
        X
    )

    # ========================================================
    # 5. Train/Test split
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(

        X,
        y_categorical,
        y_encoded
    )

    print(
        "\nTraining samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    # ========================================================
    # 6. Normalize
    # ========================================================

    (
        X_train,
        X_test,
        mean,
        std
    ) = normalize_features(

        X_train,
        X_test
    )

    # ========================================================
    # 7. Build model
    # ========================================================

    model = build_model()

    model.summary()

    # ========================================================
    # 8. Callbacks
    # ========================================================

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1
    )

    checkpoint = ModelCheckpoint(

        BEST_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    )

    # ========================================================
    # 9. Train
    # ========================================================

    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_test,
            y_test
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping,
            checkpoint
        ],

        verbose=1
    )

    # ========================================================
    # 10. Save preprocessing information
    # ========================================================

    np.save(
        "models/classes.npy",
        label_encoder.classes_
    )

    np.savez(
        "models/normalization.npz",

        mean=mean,

        std=std
    )

    # ========================================================
    # 11. Save test data
    # ========================================================

    np.save(
        "models/X_test.npy",
        X_test
    )

    np.save(
        "models/y_test.npy",
        y_test
    )

    print(
        "\nTraining completed."
    )

    print(
        "Best model:",
        BEST_MODEL_PATH
    )


if __name__ == "__main__":

    main()
