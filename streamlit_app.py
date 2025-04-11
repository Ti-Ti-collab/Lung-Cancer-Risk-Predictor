import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
with open("lung_disease_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Lung Cancer Risk Prediction App")

def encode(value):
    return 1 if value in ['Yes', 'Male'] else 0

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100)
smoking = st.selectbox("Smoking", ["Yes", "No"])
finger_discoloration = st.selectbox("Finger Discoloration", ["Yes", "No"])
mental_stress = st.selectbox("Mental Stress", ["Yes", "No"])
exposure_to_pollution = st.selectbox("Exposure to Pollution", ["Yes", "No"])
long_term_illness = st.selectbox("Long Term Illness", ["Yes", "No"])
immune_weakness = st.selectbox("Immune Weakness", ["Yes", "No"])
stress_immune = st.selectbox("Stress Immune", ["Yes", "No"])
smoking_family_history = st.selectbox("Smoking Family History", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
alcohol_consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"])
chest_tightness = st.selectbox("Chest Tightness", ["Yes", "No"])
throat_discomfort = st.selectbox("Throat Discomfort", ["Yes", "No"])
breathing_issue = st.selectbox("Breathing Issue", ["Yes", "No"])
energy_level = st.slider("Energy Level", 1, 10)
oxygen_saturation = st.slider("Oxygen Saturation", 70, 100)

# DataFrame
input_data = pd.DataFrame({
    "GENDER": [encode(gender)],
    "AGE": [age],
    "SMOKING": [encode(smoking)],
    "FINGER_DISCOLORATION": [encode(finger_discoloration)],
    "MENTAL_STRESS": [encode(mental_stress)],
    "EXPOSURE_TO_POLLUTION": [encode(exposure_to_pollution)],
    "LONG_TERM_ILLNESS": [encode(long_term_illness)],
    "IMMUNE_WEAKNESS": [encode(immune_weakness)],
    "STRESS_IMMUNE": [encode(stress_immune)],
    "SMOKING_FAMILY_HISTORY": [encode(smoking_family_history)],
    "FAMILY_HISTORY": [encode(family_history)],
    "ALCOHOL_CONSUMPTION": [encode(alcohol_consumption)],
    "CHEST_TIGHTNESS": [encode(chest_tightness)],
    "THROAT_DISCOMFORT": [encode(throat_discomfort)],
    "BREATHING_ISSUE": [encode(breathing_issue)],
    "ENERGY_LEVEL": [energy_level],
    "OXYGEN_SATURATION": [oxygen_saturation],
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of class '1'

    result = "High Risk" if prediction == 1 else "Low Risk"
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {probability * 100:.2f}%")









