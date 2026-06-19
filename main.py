"""Run the complete Iris classification project pipeline."""

from pathlib import Path
import json
import joblib

from src.data_loader import load_iris_data
from src.preprocess import split_and_scale
from src.model import find_best_k, train_knn
from src.evaluate import evaluate_model, save_confusion_matrix, save_k_scores


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Loading Iris dataset...")
    features, target, class_names = load_iris_data()

    print("Dataset shape:", features.shape)
    print("Classes:", ", ".join(class_names))

    x_train, x_test, y_train, y_test, scaler = split_and_scale(features, target)

    print("Finding best K value...")
    best_k, k_scores = find_best_k(x_train, x_test, y_train, y_test, max_k=20)
    print(f"Best K selected: {best_k}")

    print("Training final model...")
    model = train_knn(x_train, y_train, k=best_k)

    print("Evaluating model...")
    results = evaluate_model(model, x_test, y_test, class_names)

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_macro']:.4f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    joblib.dump(model, OUTPUT_DIR / "knn_iris_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "iris_scaler.joblib")

    save_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        OUTPUT_DIR / "confusion_matrix.png",
    )
    save_k_scores(k_scores, OUTPUT_DIR / "k_scores.csv")

    report_path = OUTPUT_DIR / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results["classification_report"], file, indent=4)

    print("Project completed. Check the outputs folder for saved results.")


if __name__ == "__main__":
    main()
