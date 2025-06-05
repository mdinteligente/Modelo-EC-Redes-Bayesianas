import streamlit as st
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ───────────────────────────────────────
# Cargar datos desde GitHub
# ───────────────────────────────────────
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/tu_usuario/tu_repo/main/heart_dataset.csv"
    df = pd.read_csv(url)
    df = df.astype("category")
    return df

df = cargar_datos()

# ───────────────────────────────────────
# Construir red bayesiana sencilla
# ───────────────────────────────────────
modelo = BayesianNetwork([
    ("sex", "target"),
    ("cp", "target"),
    ("thal", "target")
])

modelo.fit(df, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(modelo)

# ───────────────────────────────────────
# Interfaz Streamlit
# ───────────────────────────────────────
st.title("Modelo EC con Red Bayesiana")
st.markdown("Predicción de enfermedad cardíaca según variables clínicas.")

# Entrada de usuario
sexo = st.selectbox("Sexo (0: Mujer, 1: Hombre)", ["0", "1"])
cp = st.selectbox("Tipo de dolor torácico", ["0", "1", "2", "3"])
thal = st.selectbox("Thal (0, 1, 2)", ["0", "1", "2"])

# Predicción
if st.button("Predecir"):
    resultado = infer.query(
        variables=["target"],
        evidence={"sex": sexo, "cp": cp, "thal": thal}
    )
    prob = resultado.values
    st.write(f"Probabilidad de **no enfermedad** (target=0): {prob[0]:.2f}")
    st.write(f"Probabilidad de **enfermedad cardíaca** (target=1): {prob[1]:.2f}")


