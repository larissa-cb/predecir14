import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
model = joblib.load("best_model.pkl")

# Configuración de la página
st.set_page_config(page_title="Deserción Universitaria", page_icon="🎓", layout="wide")
st.title("🎓 Sistema de Alerta Temprana para Deserción Estudiantil")
st.markdown("Este sistema utiliza un modelo **XGBoost** entrenado para predecir el riesgo de abandono académico.")

# Cargar modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# Sidebar para entrada de datos
st.sidebar.header("📋 Información del Estudiante")

curricular_2nd_approved = st.sidebar.slider("Materias 2º semestre aprobadas", 0, 10, 5)
academic_efficiency = st.sidebar.slider("Eficiencia académica (%)", 0, 100, 75)
tuition_fees_up_to_date = st.sidebar.selectbox("Matrícula al día", ["Sí", "No"])
curricular_2nd_enrolled = st.sidebar.slider("Materias 2º semestre inscritas", 0, 10, 6)
curricular_2nd_evaluations = st.sidebar.slider("Evaluaciones 2º semestre", 0, 20, 10)
educational_special_needs = st.sidebar.selectbox("Necesidades educativas especiales", ["Sí", "No"])
academic_load = st.sidebar.slider("Carga académica (ECTS)", 0, 60, 30)
scholarship_holder = st.sidebar.selectbox("Becado", ["Sí", "No"])
curricular_1st_approved = st.sidebar.slider("Materias 1º semestre aprobadas", 0, 10, 4)
curricular_1st_credited = st.sidebar.slider("Materias 1º semestre convalidadas", 0, 10, 2)

# Botón para predecir
if st.sidebar.button("🔍 Predecir Riesgo"):
    X_input = np.array([[
        curricular_2nd_approved,
        academic_efficiency / 100,
        1 if tuition_fees_up_to_date == "Sí" else 0,
        curricular_2nd_enrolled,
        curricular_2nd_evaluations,
        1 if educational_special_needs == "Sí" else 0,
        academic_load,
        1 if scholarship_holder == "Sí" else 0,
        curricular_1st_approved,
        curricular_1st_credited
    ]])

    prediction = model.predict(X_input)[0]
probabilities = model.predict_proba(X_input)[0]

    risk_labels = ["🚨 Alto Riesgo", "⚠️ Riesgo Medio", "✅ Bajo Riesgo"]
    risk_level = risk_labels[prediction]
    confidence = probabilities[prediction]

    st.subheader("📊 Resultados de la Predicción")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nivel de Riesgo", risk_level)
    with col2:
        st.metric("Confianza", f"{confidence*100:.1f}%")
    with col3:
        st.metric("Score de Riesgo", f"{max(probabilities)*100:.1f}/100")

    st.progress(probabilities[0], text=f"Probabilidad de Alto Riesgo: {probabilities[0]*100:.1f}%")

    st.subheader("📈 Distribución de Probabilidades")
    df = pd.DataFrame({
        "Categoría": risk_labels,
        "Probabilidad": [f"{p*100:.1f}%" for p in probabilities]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("📊 Importancia de Características")
    importance = {
        "Materias 2º semestre aprobadas": 0.2337,
        "Eficiencia académica (%)": 0.1854,
        "Matrícula al día": 0.0483,
        "Materias 2º semestre inscritas": 0.0481,
        "Evaluaciones 2º semestre": 0.0352,
        "Necesidades educativas especiales": 0.0278,
        "Carga académica (ECTS)": 0.0252,
        "Becado": 0.0204,
        "Materias 1º semestre aprobadas": 0.0191,
        "Materias 1º semestre convalidadas": 0.0174
    }

    fig, ax = plt.subplots()
    ax.barh(list(importance.keys()), list(importance.values()), color="skyblue")
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de Características según análisis previo")
    ax.invert_yaxis()
    st.pyplot(fig)

else:
    st.info("👈 Introduce los datos en la barra lateral y pulsa 'Predecir Riesgo'.")



