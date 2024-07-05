import streamlit as st
import pickle
import numpy as np

# Load the trained model from the pickle file
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction")

# Create input boxes for each feature
pclass = st.selectbox("Pclass", [1, 2, 3], index=0)
sex = st.selectbox("Sex", ["male", "female"], index=0)
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=0.0)

# Convert inputs to the format expected by the model
sex = 1 if sex == "male" else 0
input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    st.write(f"The predicted Survival is: {prediction[0]}")

# Run the Streamlit app: streamlit run app.py
