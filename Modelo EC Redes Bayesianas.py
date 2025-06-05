import streamlit as st
import pandas as pd
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node
import numpy as np

# ────────────────────────────────
# Cargar y preparar los datos
# ────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("https://raw.githubusercontent.com/tu_usuario/tu_repo/main/heart_dataset.csv")
    df = df.astype({
        "sex": "category", "cp": "category", "fbs": "category",
        "restecg": "category", "exang": "category", "slope": "category",
        "ca": "category", "thal": "category", "target": "category"
    })
    return df

df = cargar_datos()

# ────────────────────────────────
# Construir red bayesiana (simulada para demo)
# ────────────────────────────────
# Definir distribuciones ficticias
sex_dist = DiscreteDistribution({'0': 0.45, '1': 0.55})
cp_dist = DiscreteDistribution({'0': 0.3, '1': 0.4, '2': 0.2, '3': 0.1})
target_cpt = ConditionalProbabilityTable(
    [
        ['0', '0', 0.8],
        ['0', '1', 0.2],
        ['1', '0', 0.4],
        ['1', '1', 0.6],
    ],
    [sex_dist]
)

# Crear nodos
sex_node = Node(sex_dist, name="sex")
cp_node = Node(cp_dist, name="cp")
target_node = Node(target_cpt, name="target")

# Crear modelo
model = BayesianNetwork("Predicción de enfermedad cardíaca")
model.add_states(sex_node, cp_node, target_node)
model.add_edge(sex_node, target_node)
model.bake()

# ────────────────────────────────
# Interfaz Streamlit
# ────────────────────────────────
st.title("Modelo EC con Redes Bayesianas")
st.markdown("Predicción basada en variables clínicas.")

sexo = st.selectbox("Sexo", options=["0", "1"])
cp = st.selectbox("Tipo de dolor torácico (cp)", options=["0", "1", "2", "3"])

if st.button("Predecir probabilidad"):
    pred = model.predict_proba({'sex': sexo})
    prob = pred[2].parameters[0]
    st.write(f"Probabilidad de enfermedad cardíaca (target=1): **{prob['1']*100:.1f}%**")

