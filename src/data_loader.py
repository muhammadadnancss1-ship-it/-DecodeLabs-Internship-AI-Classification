"""Load the Iris dataset in a clean pandas format."""

from sklearn.datasets import load_iris
import pandas as pd


def load_iris_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return features, target labels, and class names for the Iris dataset."""
    iris = load_iris(as_frame=True)

    features = iris.data.copy()
    target = iris.target.copy()
    class_names = list(iris.target_names)

    return features, target, class_names
