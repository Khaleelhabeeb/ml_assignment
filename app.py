import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Disease Prediction", page_icon=":stethoscope:", layout="centered")

st.title("Disease Prediction")
st.write("Enter patient details to predict the most likely disease class.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = {
        "Gender": joblib.load("le_gender.pkl"),
        "Blood Pressure": joblib.load("le_bp.pkl"),
        "Cholesterol": joblib.load("le_chol.pkl"),
        "Glucose": joblib.load("le_gluc.pkl"),
        "Smoking": joblib.load("le_smoke.pkl"),
        "Alcohol Consumption": joblib.load("le_alc.pkl"),
        "Exercise": joblib.load("le_ex.pkl"),
        "Family History": joblib.load("le_fh.pkl"),
    }
    return model, label_encoder, scaler, encoders

model, label_encoder, scaler, encoders = load_artifacts()

with st.form("input_form"):
    st.subheader("Patient Information")

    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    blood_pressure = st.selectbox("Blood Pressure", encoders["Blood Pressure"].classes_)
    cholesterol = st.selectbox("Cholesterol", encoders["Cholesterol"].classes_)
    glucose = st.selectbox("Glucose", encoders["Glucose"].classes_)
    smoking = st.selectbox("Smoking", encoders["Smoking"].classes_)
    alcohol = st.selectbox("Alcohol Consumption", encoders["Alcohol Consumption"].classes_)
    exercise = st.selectbox("Exercise", encoders["Exercise"].classes_)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=22.0, step=0.1)
    family_history = st.selectbox("Family History", encoders["Family History"].classes_)

    submitted = st.form_submit_button("Predict")

if submitted:
    inputs = {
        "Gender": gender,
        "Blood Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Glucose": glucose,
        "Smoking": smoking,
        "Alcohol Consumption": alcohol,
        "Exercise": exercise,
        "Family History": family_history,
    }

    encoded = []
    for key, value in inputs.items():
        encoder = encoders[key]
        encoded.append(int(encoder.transform([value])[0]))

    feature_vector = np.array([
        age,
        encoded[0],
        encoded[1],
        encoded[2],
        encoded[3],
        encoded[4],
        encoded[5],
        encoded[6],
        bmi,
        encoded[7],
    ], dtype=float).reshape(1, -1)

    scaled = scaler.transform(feature_vector)
    pred_idx = model.predict(scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    st.success(f"Predicted Disease: {pred_label}")
