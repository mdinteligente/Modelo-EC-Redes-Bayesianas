# app.py
import streamlit as st
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader

# Cargar modelo bayesiano previamente guardado en formato .bif
@st.cache_resource
def load_model():
    reader = BIFReader("modelo_clinico.bif")  # El archivo .bif debe estar en el mismo directorio
    model = reader.get_model()
    return model

model = load_model()
infer = VariableElimination(model)

st.title("Predicción de Riesgo Cardiovascular con Red Bayesiana")
st.write("Ingrese los valores del paciente para estimar la probabilidad de enfermedad cardíaca.")

# Entradas del usuario
age = st.slider("Edad", 29, 77, 55)
trestbps = st.slider("Presión arterial en reposo (mm Hg)", 90, 180, 130)
chol = st.slider("Colesterol sérico (mg/dL)", 100, 600, 240)
thalach = st.slider("Frecuencia cardíaca máxima", 70, 210, 150)
oldpeak = st.slider("Depresión del ST inducida por ejercicio", 0.0, 6.2, 1.0, step=0.1)

sex = st.selectbox("Sexo", options=["0", "1"], format_func=lambda x: "Femenino" if x == "0" else "Masculino")
cp = st.selectbox("Tipo de dolor torácico (cp)", ["0", "1", "2", "3"])
fbs = st.selectbox("Glucosa en ayunas > 120 mg/dL", ["0", "1"])
restecg = st.selectbox("ECG en reposo", ["0", "1", "2"])
exang = st.selectbox("Angina inducida por ejercicio", ["0", "1"])
slope = st.selectbox("Pendiente del ST", ["0", "1", "2"])
ca = st.selectbox("Vasos principales coloreados", ["0", "1", "2", "3"])
thal = st.selectbox("Thalassemia", ["0", "1", "2", "3"])

# Evidencia para el modelo
data_evidence = {
    "age": str(age),
    "trestbps": str(trestbps),
    "chol": str(chol),
    "thalach": str(thalach),
    "oldpeak": str(oldpeak),
    "sex": sex,
    "cp": cp,
    "fbs": fbs,
    "restecg": restecg,
    "exang": exang,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}

if st.button("Estimar riesgo"):
    result = infer.query(variables=["target"], evidence=data_evidence)
    proba = result.values[1]
    st.success(f"Probabilidad estimada de enfermedad cardíaca: {proba:.2%}")
    st.info("Nota: Esta herramienta es educativa y no reemplaza el juicio clínico.")