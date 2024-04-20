#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
def load_data():
    data = pd.read_csv("soil_measures.csv")
    return data

data = load_data()

# Splitting data into features and target
X = data.drop(columns='crop')
y = data['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Logistic Regression model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)

# Streamlit interface
st.title('Crop Selection Assistant')
st.write("This tool helps farmers select the best crop based on soil measurements.")

# User input parameters
st.sidebar.header('Input Soil Parameters')
def user_input_features():
    N = st.sidebar.slider('Nitrogen level (N)', float(X['N'].min()), float(X['N'].max()), float(X['N'].mean()))
    P = st.sidebar.slider('Phosphorous level (P)', float(X['P'].min()), float(X['P'].max()), float(X['P'].mean()))
    K = st.sidebar.slider('Potassium level (K)', float(X['K'].min()), float(X['K'].max()), float(X['K'].mean()))
    ph = st.sidebar.slider('Soil ph', float(X['ph'].min()), float(X['ph'].max()), float(X['ph'].mean()))
    return pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'ph': [ph]})

input_df = user_input_features()

# Display the input features
st.subheader('Specified Input parameters')
st.write(input_df)

# Predict and display the output
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Predicted Crop')
st.write(prediction[0])
st.write('Prediction Probability')
st.write(prediction_proba)

# Optional: Display pie chart of crops
st.subheader('Crop Distribution in Dataset')
if st.button('Show Pie Chart'):
    fig, ax = plt.subplots()
    data['crop'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)
