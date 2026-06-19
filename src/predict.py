"""Make predictions using a saved model and scaler."""

import joblib


MODEL_PATH = "outputs/knn_iris_model.joblib"
SCALER_PATH = "outputs/iris_scaler.joblib"


def predict_flower(sample: list[float]) -> str:
    """Predict the Iris class for one flower sample.

    Expected order:
    [sepal length, sepal width, petal length, petal width]
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    class_names = ["setosa", "versicolor", "virginica"]
    scaled_sample = scaler.transform([sample])
    prediction = model.predict(scaled_sample)[0]

    return class_names[prediction]


if __name__ == "__main__":
    sample_flower = [5.1, 3.5, 1.4, 0.2]
    result = predict_flower(sample_flower)
    print(f"Predicted class: {result}")
