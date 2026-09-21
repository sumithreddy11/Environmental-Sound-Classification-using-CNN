# ============================================================
# evaluate.py
# Model evaluation
# ============================================================

import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from config import (
    BEST_MODEL_PATH,
    RESULTS_DIR
)


def main():

    # ========================================================
    # 1. Load model
    # ========================================================

    model = load_model(
        BEST_MODEL_PATH
    )

    # ========================================================
    # 2. Load test data
    # ========================================================

    X_test = np.load(
        "models/X_test.npy"
    )

    y_test = np.load(
        "models/y_test.npy"
    )

    classes = np.load(
        "models/classes.npy"
    )

    # ========================================================
    # 3. Predictions
    # ========================================================

    probabilities = model.predict(
        X_test
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    y_true = np.argmax(
        y_test,
        axis=1
    )

    # ========================================================
    # 4. Accuracy
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ========================================================
    # 5. Classification report
    # ========================================================

    report = classification_report(

        y_true,

        y_pred,

        target_names=classes
    )

    print(
        "\nClassification Report\n"
    )

    print(
        report
    )

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:

        file.write(report)

    # ========================================================
    # 6. Confusion matrix
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=classes,

        yticklabels=classes
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    main()
