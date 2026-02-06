import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.write("App Started")
st.title("🍷 Wine Quality Prediction App")

df = pd.read_csv("winequality-red.csv")

st.write("Dataset Loaded")

# Create Label (Good / Bad Wine)
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(['quality','quality_label'], axis=1)
y = df['quality_label']

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

st.write("Model Trained Successfully")

# Sidebar Inputs
st.sidebar.header("Enter Wine Properties")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 15.0, 7.0)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 15.0, 2.5)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.08)
free_sulfur = st.sidebar.slider("Free Sulfur Dioxide", 1, 80, 15)
total_sulfur = st.sidebar.slider("Total Sulfur Dioxide", 10, 300, 46)
density = st.sidebar.slider("Density", 0.990, 1.005, 0.996)
ph = st.sidebar.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.sidebar.slider("Sulphates", 0.2, 2.0, 0.6)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)

input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                         residual_sugar, chlorides, free_sulfur,
                         total_sulfur, density, ph, sulphates, alcohol]])

# Prediction
if st.button("Predict Wine Quality"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("❌ Bad Quality Wine")
