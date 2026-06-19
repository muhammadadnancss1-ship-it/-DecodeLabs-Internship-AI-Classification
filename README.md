1 Data Classification Using AI

A beginner-friendly machine learning project that builds a complete classification pipeline using the Iris dataset and the K-Nearest Neighbors algorithm.

This project follows the Project 2 requirement from the DecodeLabs Artificial Intelligence Industrial Training Kit: load a small dataset, split it into training and testing sets, apply a simple classification algorithm, and validate the result using proper metrics.

2 Project Goal

The goal of this project is to train a supervised learning model that can classify Iris flowers into one of three categories:

Setosa
Versicolor
Virginica
The model learns from flower measurements and then predicts the class of unseen flower samples.

3 What This Project Covers

This project includes the full basic AI classification workflow:

Loading and understanding the dataset
Preparing input features and target labels
Splitting data into training and testing sets
Scaling numeric features using StandardScaler
Training a K-Nearest Neighbors classifier
Testing multiple K values
Selecting the best K value
Evaluating the final model
Saving the trained model
Making predictions on new flower data
4 Dataset Used

This project uses the built-in Iris dataset from scikit-learn.

The dataset contains:

| Item | Value |

| Total samples | 150 | | Classes | 3 | | Features | 4 |

5 Features

Sepal length
Sepal width
Petal length
Petal width
6 Algorithm Used

K-Nearest Neighbors
K-Nearest Neighbors, also called KNN, is a simple supervised learning algorithm. It classifies a new data point by checking the nearest existing data points and using majority voting.

For example, if most of the nearest neighbors belong to setosa, the model predicts the new flower as setosa.

7 Installation

Clone the repository
2. Create a virtual environment
For Windows:

3. Install dependencies
pip install -r requirements.txt
8 How to Run the Project

Run the complete training and evaluation pipeline:

python main.py
After running the project, the outputs folder will contain:

knn_iris_model.joblib
iris_scaler.joblib
confusion_matrix.png
k_scores.csv
classification_report.json
9 How to Make a Prediction

After training the model, run:

python -m src.predict
The sample prediction uses this input:

[5.1, 3.5, 1.4, 0.2]
The order of values is:

[sepal length, sepal width, petal length, petal width]
10 Model Evaluation

The project evaluates the model using:

Accuracy
F1 score
Confusion matrix
Classification report
Accuracy is useful, but it is not the only metric. The F1 score and confusion matrix give a better view of how well the model performs for each class.

11 Why Feature Scaling Is Used

KNN depends on distance between data points. If one feature has a larger numeric range than another, it can wrongly dominate the result.

To avoid this problem, the project uses StandardScaler, which balances the feature values before training.

12 Learning Outcome

By completing this project, you will understand how a basic supervised learning pipeline works in real AI projects.

You will learn how to:

Work with tabular data
Prepare training and testing data
Train a classification model
Tune model parameters
Evaluate model performance
Save and reuse a trained model
13 Future Improvements

This project can be improved by adding:

Logistic Regression comparison
Decision Tree comparison
Cross-validation
A small Streamlit web app
More visual charts
User input form for predictions
