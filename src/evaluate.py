"""Evaluate the trained classification model."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_model(model, x_test, y_test, class_names: list[str]) -> dict:
    """Return the main classification metrics."""
    predictions = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=class_names,
            output_dict=True,
        ),
        "predictions": predictions,
    }


def save_confusion_matrix(matrix, class_names: list[str], path: str) -> None:
    """Save a simple confusion matrix chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_k_scores(scores: list[dict], path: str) -> None:
    """Save K tuning scores to a CSV file."""
    pd.DataFrame(scores).to_csv(path, index=False)
