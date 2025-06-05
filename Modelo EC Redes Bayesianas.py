import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# Cargar modelo entrenado
model = joblib.load("modelo_xgb.pkl")

# Título de la aplicación
st.title("Predicción de Enfermedad Cardiovascular con XGBoost")

# Ingreso de datos por el usuario
st.sidebar.header("Ingrese los datos del paciente")

def user_input_features():
    age = st.sidebar.slider("Edad", 29, 77, 50)
    trestbps = st.sidebar.slider("Presión arterial en reposo", 90, 180, 130)
    chol = st.sidebar.slider("Colesterol sérico", 120, 560, 240)
    thalach = st.sidebar.slider("Frecuencia cardíaca máxima", 70, 210, 150)
    oldpeak = st.sidebar.slider("Depresión ST", 0.0, 6.5, 1.0)
    sex = st.sidebar.selectbox("Sexo", options=["Masculino", "Femenino"])
    cp = st.sidebar.selectbox("Tipo de dolor torácico", ["Angina típica", "Angina atípica", "No angina", "Asintomático"])
    fbs = st.sidebar.selectbox("Glicemia en ayuno > 120 mg/dl", ["No", "Sí"])
    restecg = st.sidebar.selectbox("ECG en reposo", ["Normal", "Anormalidad ST-T", "Hipertrofia ventricular"])
    exang = st.sidebar.selectbox("Angina inducida por ejercicio", ["No", "Sí"])
    slope = st.sidebar.selectbox("Pendiente del ST", ["Ascendente", "Plana", "Descendente"])
    ca = st.sidebar.selectbox("Nº de vasos coloreados", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thal", ["Normal", "Defecto fijo", "Defecto reversible"])

    data = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "sex": 1 if sex == "Masculino" else 0,
        "cp": ["Angina típica", "Angina atípica", "No angina", "Asintomático"].index(cp),
        "fbs": 1 if fbs == "Sí" else 0,
        "restecg": ["Normal", "Anormalidad ST-T", "Hipertrofia ventricular"].index(restecg),
        "exang": 1 if exang == "Sí" else 0,
        "slope": ["Ascendente", "Plana", "Descendente"].index(slope),
        "ca": ca,
        "thal": ["Normal", "Defecto fijo", "Defecto reversible"].index(thal),
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Predicción
prediction_proba = model.predict_proba(input_df)[0, 1]
prediction_label = "Enfermedad cardiovascular probable" if prediction_proba > 0.5 else "Sin enfermedad cardiovascular"

st.subheader("Resultado de la predicción")
st.write(f"Probabilidad: **{prediction_proba:.2f}** → **{prediction_label}**")

# Gráfico de importancia de variables
st.subheader("Importancia de las variables (global)")
importance = model.feature_importances_
features = model.get_booster().feature_names if model.get_booster().feature_names else input_df.columns
importancia_df = pd.DataFrame({'feature': features, 'importance': importance})
importancia_df = importancia_df.sort_values(by='importance', ascending=True)

fig, ax = plt.subplots()
ax.barh(importancia_df['feature'], importancia_df['importance'], color='skyblue')
ax.set_xlabel('Importancia')
st.pyplot(fig)




