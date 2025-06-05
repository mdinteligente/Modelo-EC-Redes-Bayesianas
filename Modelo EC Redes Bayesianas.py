import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
modelo = joblib.load("modelo_xgb.pkl")

# Variables esperadas
input_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex', 'cp', 'fbs',
                 'restecg', 'exang', 'slope', 'ca', 'thal']

st.title("Predicción de Enfermedad Cardiovascular")

st.markdown("Completa los datos del paciente:")

# Entradas numéricas
age = st.slider("Edad", 29, 77, 55)
trestbps = st.slider("Presión arterial en reposo (mmHg)", 90, 200, 120)
chol = st.slider("Colesterol sérico (mg/dL)", 100, 400, 200)
thalach = st.slider("Frecuencia cardíaca máxima", 70, 210, 150)
oldpeak = st.slider("Oldpeak (ST depresion)", 0.0, 6.0, 1.0, step=0.1)

# Entradas categóricas
sex = st.selectbox("Sexo", ['0', '1'])
cp = st.selectbox("Tipo de dolor torácico (cp)", ['0', '1', '2', '3'])
fbs = st.selectbox("Glucemia en ayunas >120 mg/dL", ['0', '1'])
restecg = st.selectbox("Resultados electrocardiográficos en reposo", ['0', '1', '2'])
exang = st.selectbox("Angina inducida por ejercicio", ['0', '1'])
slope = st.selectbox("Pendiente del ST", ['0', '1', '2'])
ca = st.selectbox("Número de vasos coloreados con fluoroscopia", ['0', '1', '2', '3'])
thal = st.selectbox("Talasemia", ['1', '2', '3'])

# Botón de predicción
if st.button("Predecir"):
    datos_usuario = pd.DataFrame([[age, trestbps, chol, thalach, oldpeak, sex, cp, fbs,
                                    restecg, exang, slope, ca, thal]], columns=input_columns)
    datos_usuario = datos_usuario.astype(modelo.get_booster().feature_types)
    prob = modelo.predict_proba(datos_usuario)[0][1]
    pred = modelo.predict(datos_usuario)[0]

    st.markdown(f"### Probabilidad de enfermedad cardiovascular: **{prob:.2%}**")
    st.markdown(f"### Clasificación: {'🟥 Positiva' if pred == 1 else '🟩 Negativa'}")

    # Importancia de variables
    st.markdown("#### Importancia de variables en el modelo")
    importancias = modelo.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(input_columns, importancias)
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de cada variable")
    st.pyplot(fig)





