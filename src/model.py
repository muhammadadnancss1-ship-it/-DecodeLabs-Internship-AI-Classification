"""Train and tune a K-Nearest Neighbors classifier."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def train_knn(x_train, y_train, k: int = 5) -> KNeighborsClassifier:
    """Train a KNN model with the selected value of K."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    return model


def find_best_k(x_train, x_test, y_train, y_test, max_k: int = 20) -> tuple[int, list[dict]]:
    """Try different K values and return the best one by test accuracy."""
    scores = []

    for k in range(1, max_k + 1):
        model = train_knn(x_train, y_train, k=k)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append({"k": k, "accuracy": accuracy, "error_rate": 1 - accuracy})

    best_row = max(scores, key=lambda item: item["accuracy"])
    return best_row["k"], scores
