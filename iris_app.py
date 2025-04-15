import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
st.sidebar.title("Model Configuration")
model_name = st.sidebar.selectbox("Select Classifier", ("Random Forest", "Logistic Regression", "KNN"))
input_mode = st.sidebar.radio("Input Mode", ("Slider", "Manual"))
if st.sidebar.checkbox("Show raw dataset"):
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = df['target'].apply(lambda i: target_names[i])
    st.subheader("Raw Iris Dataset")
    st.write(df.head())
if model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
else:
    model = KNeighborsClassifier()

model.fit(X, y)

st.title("Iris Flower Species Classifier")
st.subheader("Input Flower Measurements")

def get_input():
    if input_mode == "Slider":
        sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
        sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
        petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.35)
        petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)
    else:
        sepal_length = st.number_input('Sepal Length (cm)', value=5.8)
        sepal_width = st.number_input('Sepal Width (cm)', value=3.0)
        petal_length = st.number_input('Petal Length (cm)', value=4.35)
        petal_width = st.number_input('Petal Width (cm)', value=1.3)

    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

input_features = get_input()

prediction = model.predict(input_features)[0]
prediction_proba = model.predict_proba(input_features)[0]

st.subheader("Prediction")
st.write(f"Predicted Species: **{target_names[prediction]}**")

st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame([prediction_proba], columns=target_names)
st.bar_chart(proba_df.T)

if st.checkbox("Show model performance on entire dataset"):
    y_pred_all = model.predict(X)
    cm = confusion_matrix(y, y_pred_all)
    cr = classification_report(y, y_pred_all, target_names=target_names, output_dict=True)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(cr).transpose())

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=target_names, yticklabels=target_names, fmt="d", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
