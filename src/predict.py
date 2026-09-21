# ============================================================
# predict.py
# Prediction on unseen audio
# ============================================================

import sys
import numpy as np

from tensorflow.keras.models import load_model

from feature_extraction import extract_mfcc

from config import BEST_MODEL_PATH


def predict_audio(file_path):

    # ========================================================
    # Load model
    # ========================================================

    model = load_model(
        BEST_MODEL_PATH
    )

    # ========================================================
    # Load classes
    # ========================================================

    classes = np.load(
        "models/classes.npy"
    )

    # ========================================================
    # Load normalization parameters
    # ========================================================

    normalization = np.load(
        "models/normalization.npz"
    )

    mean = normalization[
        "mean"
    ]

    std = normalization[
        "std"
    ]

    # ========================================================
    # Extract MFCC
    # ========================================================

    mfcc = extract_mfcc(
        file_path
    )

    if mfcc is None:

        print(
            "Could not process audio file."
        )

        return

    # ========================================================
    # Prepare input
    # ========================================================

    sample = mfcc[
        np.newaxis,
        ...,
        np.newaxis
    ]

    # ========================================================
    # Normalize
    # ========================================================

    sample = (
        sample - mean
    ) / (
        std + 1e-8
    )

    # ========================================================
    # Prediction
    # ========================================================

    probabilities = model.predict(
        sample,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ]

    # ========================================================
    # Display results
    # ========================================================

    print(
        "\n"
        + "=" * 50
    )

    print(
        "Prediction Probabilities"
    )

    print(
        "=" * 50
    )

    for class_name, probability in zip(
        classes,
        probabilities
    ):

        print(
            f"{class_name:20s}: "
            f"{probability * 100:.2f}%"
        )

    print(
        "\nFinal Prediction"
    )

    print(
        "----------------"
    )

    print(
        f"Class      : "
        f"{predicted_class}"
    )

    print(
        f"Confidence : "
        f"{confidence * 100:.2f}%"
    )

    if confidence < 0.60:

        print(
            "\n⚠ Low confidence prediction."
        )

    else:

        print(
            "\n✓ Relatively confident prediction."
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python src/predict.py "
            "path/to/audio.wav"
        )

        sys.exit(1)

    audio_path = sys.argv[1]

    predict_audio(
        audio_path
    )
