#  Iris Species Classifier Web App

This is a simple interactive web application built with **Streamlit** for classifying Iris flower species using various machine learning models. Users can choose between Random Forest, Logistic Regression, or K-Nearest Neighbors to predict the species based on flower measurements.

##  Features

- Load and explore the classic Iris dataset.
- Choose a classifier: Random Forest, Logistic Regression, or KNN.
- Enter flower measurements manually or with sliders.
- View prediction results and probabilities.
- Visualize classification report and confusion matrix.

##  Getting Started

### 1. Clone the repository

git clone https://github.com/RaviKiran752/Iris-Flower-Classifier.git
cd iris-classifier-app

2. Install dependencies

It's recommended to use a virtual environment.

pip install -r requirements.txt

3. Run the app

streamlit run iris_app.py

 Model Details

This app uses scikit-learn classifiers trained on the full Iris dataset. No model persistence is used—models are trained each time the app is launched.
 Requirements

See requirements.txt.
 Screenshots

Add screenshots here if needed.
 Dataset

    Iris dataset from scikit-learn’s built-in datasets.

 Tech Stack

    Python

    Streamlit

    scikit-learn

    pandas, numpy

    seaborn, matplotlib# Iris-Flower-Classifier
