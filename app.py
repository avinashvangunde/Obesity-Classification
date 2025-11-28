import streamlit as st
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline  

best_pipe = pickle.load(open("best_obesity_model.pkl", "rb"))

st.title("Obesity Classification")

st.write("Enter your information to estimate whether you are obese or not. "
         "This demo is for educational purposes only, not medical advice.")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 14, 60, 25)
height = st.number_input("Height (meters)", min_value=1.3, max_value=2.1, value=1.70, step=0.01)
weight = st.number_input("Weight (kg)", min_value=35.0, max_value=180.0, value=70.0, step=0.5)
family_history = st.selectbox("Family history with overweight", ["yes", "no"])
favc = st.selectbox("Frequent high caloric food (FAVC)", ["yes", "no"])
fcvc = st.selectbox("Vegetable consumption frequency (FCVC)", [1, 2, 3])
ncp = st.selectbox("Number of main meals (NCP)", [1, 2, 3, 4])
caec = st.selectbox("Food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Smoke", ["yes", "no"])
ch2o = st.selectbox("Daily water (CH2O)", [1, 2, 3])
scc = st.selectbox("Calories monitoring (SCC)", ["yes", "no"])
faf = st.selectbox("Physical activity (FAF)", [0, 1, 2, 3])
tue = st.selectbox("Time using tech devices (TUE)", [0, 1, 2])
calc = st.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Main transport (MTRANS)", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

if st.button("Predict"):
    input_dict = {
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [family_history],
        "FAVC": [favc],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CAEC": [caec],
        "SMOKE": [smoke],
        "CH2O": [ch2o],
        "SCC": [scc],
        "FAF": [faf],
        "TUE": [tue],
        "CALC": [calc],
        "MTRANS": [mtrans],
    }

    input_df = pd.DataFrame(input_dict)
    pred = best_pipe.predict(input_df)[0]
    proba = best_pipe.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"Model prediction: Obese (probability {proba:.2f})")
    else:
        st.success(f"Model prediction: Not obese (probability {proba:.2f})")

