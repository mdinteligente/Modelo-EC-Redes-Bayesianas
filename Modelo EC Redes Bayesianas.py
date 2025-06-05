# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Cargar modelo y preprocesadores
# ─────────────────────────────────────────────
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")  # Si usaste OneHotEncoding

# ─────────────────────────────────────────────
# Función para recolectar entrada del usuario
# ─────────────────────────────────────────────
def user_input():
    st.sidebar.header("Datos del paciente")

    age = st.sidebar.slider("Edad", 20, 100, 50)
    trestbps = st.sidebar.slider("Presión en reposo (mmHg)", 90, 200, 120)
    chol = st.sidebar.slider("Colesterol (mg/dL)", 100, 600, 240)
    thalach = st.sidebar.slider("Frecuencia cardiaca máxima", 60, 220, 150)
    oldpeak = st.sidebar.slider("Oldpeak (ST depresión)", 0.0, 6.0, 1.0)

    sex = st.sidebar.selectbox("Sexo", ["0", "1"])
    cp = st.sidebar.selectbox("Tipo de dolor torácico", ["0", "1", "2", "3"])
    fbs = st.sidebar.selectbox("Glicemia en ayuno >120 mg/dL", ["0", "1"])
    restecg = st.sidebar.selectbox("ECG en reposo", ["0", "1", "2"])
    exang = st.sidebar.selectbox("Angina inducida por ejercicio", ["0", "1"])
    slope = st.sidebar.selectbox("Pendiente del ST", ["0", "1", "2"])
    ca = st.sidebar.selectbox("N° vasos coloreados", ["0", "1", "2", "3"])
    thal = st.sidebar.selectbox("Thal", ["0", "1", "2", "3"])

    data = {
        "age": float(age), "trestbps": float(trestbps), "chol": float(chol),
        "thalach": float(thalach), "oldpeak": float(oldpeak),
        "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg,
        "exang": exang, "slope": slope, "ca": ca, "thal": thal
    }

    return pd.DataFrame([data])

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
st.title("🫀 Predicción de Enfermedad Cardiovascular con XGBoost + SHAP")

input_df = user_input()

# Codificar variables categóricas
X_encoded = encoder.transform(input_df)
X_scaled = scaler.transform(X_encoded)

# Predicción
proba = xgb_model.predict_proba(X_scaled)[0][1]
st.subheader("📊 Probabilidad de enfermedad cardiovascular")
st.write(f"**{proba*100:.2f}%**")

# Interpretabilidad con SHAP
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_scaled)

st.subheader("🔍 Explicación de la predicción con SHAP")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=10)
st.pyplot(fig)

# SHAP bar summary
st.subheader("📈 Contribuciones globales de las variables")
fig2, ax2 = plt.subplots()
shap.plots.bar(shap_values, max_display=10)
st.pyplot(fig2)



