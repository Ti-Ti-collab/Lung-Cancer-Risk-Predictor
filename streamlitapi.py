import streamlit as st
import pickle
import pandas as pd

# Load trained model pipeline
with open("lung_disease_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# App Title
st.title("🫁 Lung Cancer Risk Predictor")

st.markdown("""
This app predicts whether an individual is at risk of developing lung cancer based on symptoms and lifestyle.
""")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Patient Information")

    age = st.slider("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety", ["Yes", "No"])
    chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    coughing = st.selectbox("Coughing", ["Yes", "No"])
    shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
    oxygen_saturation = st.slider("Oxygen Saturation (%)", 50, 100, 95)
    energy_level = st.slider("Energy Level (%)", 0, 100, 50)

    submit = st.form_submit_button("Predict")

# Mapping categorical responses
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

if submit:
    input_data = pd.DataFrame([{
        'AGE': age,
        'GENDER': binary_map[gender],
        'SMOKING': binary_map[smoking],
        'YELLOW_FINGERS': binary_map[yellow_fingers],
        'ANXIETY': binary_map[anxiety],
        'CHRONIC_DISEASE': binary_map[chronic_disease],
        'FATIGUE': binary_map[fatigue],
        'WHEEZING': binary_map[wheezing],
        'COUGHING': binary_map[coughing],
        'SHORTNESS_OF_BREATH': binary_map[shortness_of_breath],
        'OXYGEN_SATURATION': oxygen_saturation,
        'ENERGY_LEVEL': energy_level
    }])

    prediction = model.predict(input_data)[0]
    result = "✅ Low Risk" if prediction == 0 else "⚠️ High Risk"

    st.subheader("Prediction Result")
    st.success(result)






