
# ============================================================
# model.py
# CNN model architecture
# ============================================================

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

from config import (
    N_MFCC,
    MAX_LEN,
    NUM_CLASSES
)


def build_model():
    """
    Build and compile the CNN classifier.
    """

    model = Sequential([

        # ----------------------------------------------------
        # Convolution Block 1
        # ----------------------------------------------------

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(
                N_MFCC,
                MAX_LEN,
                1
            )
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Dropout(0.30),

        # ----------------------------------------------------
        # Convolution Block 2
        # ----------------------------------------------------

        Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Dropout(0.30),

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.50),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model
