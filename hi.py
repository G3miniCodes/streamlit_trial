#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Iris Flower Species Prediction")

st.write("""
This app uses a logistic regression model to predict the species of an Iris flower based on its features.
""")

st.write(f"Model Accuracy: {accuracy:.2f}")

# Input features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                          columns=iris.feature_names)

# Prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader("Input Features")
st.write(input_data)

st.subheader("Prediction")
st.write(iris.target_names[prediction][0])

st.subheader("Prediction Probability")
st.write(prediction_proba)


# In[ ]:




