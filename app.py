import streamlit as st
import re
import joblib
from PIL import Image
from googletrans import Translator
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import BytesIO
# Configurar la página
st.set_page_config(page_title="Detector de SPAM", layout="wide")

# Cargar logo
logo = Image.open("env/logo.png")

# Definir los modelos disponibles (puedes mover esto a un archivo aparte si prefieres)
MODELOS_DISPONIBLES = {
    "Multinomial Naive Bayes": "MultinomialNB",
    "SVC (Sigmoid)": "SVC",
    "KNN": "KNeighborsClassifier",
    "Logistic Regression (L1)": "LogisticRegression",
    "Decision Tree": "DecisionTreeClassifier",
    "Random Forest": "RandomForestClassifier"
}

# Inicializar traductor
traductor = Translator()

# Inicializar contadores de sesión e historial
if 'contador_spam' not in st.session_state:
    st.session_state.contador_spam = 0
if 'contador_nospam' not in st.session_state:
    st.session_state.contador_nospam = 0
if 'historial' not in st.session_state:
    st.session_state.historial = []
if 'modelo_seleccionado' not in st.session_state:
    st.session_state.modelo_seleccionado = "Multinomial Naive Bayes"

# Estilos personalizados
st.markdown("""
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #1F4172;
        }
        .subtitle {
            font-size: 18px;
            color: #4F4F4F;
        }
        .result {
            font-size: 20px;
            font-weight: bold;
        }
        .spam {
            color: #C62828;
        }
        .nospam {
            color: #2E7D32;
        }
        .keywords {
            font-size: 16px;
            margin-top: 10px;
            color: #1E88E5;
        }
        .riesgo {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Encabezado con logo
col1, col2 = st.columns([50, 100])
with col1:
    st.image(logo, width=200)
with col2:
    st.markdown("<div class='title'>🧠 Detector Inteligente de SPAM</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Escuela Profesional de Ingeniería de Sistemas de la UNSCH, Ayacucho  2025</div>", unsafe_allow_html=True)

# Selección de modelo
st.session_state.modelo_seleccionado = st.selectbox(
    "🔧 Seleccione el algoritmo de clasificación:",
    options=list(MODELOS_DISPONIBLES.keys()),
    index=list(MODELOS_DISPONIBLES.keys()).index(st.session_state.modelo_seleccionado)
)

# Cargar modelo y vectorizador
def cargar_modelo(nombre_modelo):
    modelo_path = f"modelos/{MODELOS_DISPONIBLES[nombre_modelo]}.pkl"
    vectorizador_path = "modelos/vectorizador.pkl"
    
    if os.path.exists(modelo_path) and os.path.exists(vectorizador_path):
        try:
            modelo = joblib.load(modelo_path)
            vectorizador = joblib.load(vectorizador_path)
            return modelo, vectorizador, True
        except:
            return None, None, False
    return None, None, False

modelo, vectorizador, cargado = cargar_modelo(st.session_state.modelo_seleccionado)

if not cargado:
    st.warning(f"⚠️ El modelo {st.session_state.modelo_seleccionado} no está disponible. Usando Multinomial Naive Bayes por defecto.")
    modelo, vectorizador, _ = cargar_modelo("Multinomial Naive Bayes")

# Entrada del mensaje
mensaje_es = st.text_area("📋 Pega tu mensaje lo analizamos como SPAM o NO SPAM:", height=200)

# Botón de análisis
if st.button("✅ Comprobar"):
    st.markdown("---")

    # Extraer correos
    correos = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', mensaje_es)
    if correos:
        st.success(f"📧 Correos detectados: {', '.join(correos)}")
    else:
        st.warning("⚠️ No se detectaron correos electrónicos.")

    # Traducir mensaje al inglés
    try:
        resultado_traduccion = traductor.translate(mensaje_es, src='es', dest='en')
        mensaje_en = resultado_traduccion.text

        # Clasificación SPAM
        vector = vectorizador.transform([mensaje_en])
        prediccion = modelo.predict(vector)[0]
        probabilidades = modelo.predict_proba(vector)[0]
        confianza = round(max(probabilidades) * 100, 2)

        # Nivel de riesgo según confianza
        if confianza >= 80:
            nivel_riesgo = "ALTO"
            color_riesgo = "#C62828"
        elif confianza >= 50:
            nivel_riesgo = "INTERMEDIO"
            color_riesgo = "#F9A825"
        else:
            nivel_riesgo = "BAJO"
            color_riesgo = "#2E7D32"

        # Palabras clave comunes de SPAM
        palabras_spam = ["gratis", "urgente", "clic", "promoción", "dinero", "premio", "haz clic", "ganaste"]
        palabras_detectadas = [p for p in palabras_spam if p in mensaje_es.lower()]

        # Mostrar resultado y confianza
        if prediccion == 1:
            st.session_state.contador_spam += 1
            st.markdown(f"<div class='result spam'>🚫 Este mensaje es SPAM (Confianza: {confianza}%)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='keywords'>🔍 Modelo usado: {st.session_state.modelo_seleccionado}</div>", unsafe_allow_html=True)
        else:
            st.session_state.contador_nospam += 1
            st.markdown(f"<div class='result nospam'>✅ Este mensaje es NO SPAM (Confianza: {confianza}%)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='keywords'>🔍 Modelo usado: {st.session_state.modelo_seleccionado}</div>", unsafe_allow_html=True)

        # Mostrar palabras clave detectadas
        if palabras_detectadas:
            st.markdown("<div class='keywords'>🔍 Palabras clave detectadas: " + ", ".join(palabras_detectadas) + "</div>", unsafe_allow_html=True)
        
        ### 🧩 Recomendaciones basadas en el análisis
        if nivel_riesgo == "ALTO":
            st.warning("🔺 Este mensaje contiene indicadores claros de SPAM. Evita responder o hacer clic en enlaces. Informa al administrador si se trata de un correo institucional.")
        elif nivel_riesgo == "INTERMEDIO":
            st.info("⚠️ Este mensaje tiene algunas señales sospechosas. Revisa con cautela antes de responder.")
        else:
            st.success("✅ El mensaje no presenta riesgos evidentes. Puedes proceder con normalidad.")

        # Mostrar nivel de riesgo
        st.markdown(f"<div class='riesgo' style='color: {color_riesgo};'>🔒 Nivel de riesgo: {nivel_riesgo}</div>", unsafe_allow_html=True)

        # Agregar al historial
        st.session_state.historial.append({
            "Modelo": st.session_state.modelo_seleccionado,
            "Mensaje (ES)": mensaje_es,
            "Mensaje (EN)": mensaje_en,
            "Correos": ", ".join(correos) if correos else "Ninguno",
            "Resultado": "SPAM" if prediccion == 1 else "NO SPAM",
            "Confianza": confianza,
            "Nivel de Riesgo": nivel_riesgo,
            "Palabras clave": ", ".join(palabras_detectadas) if palabras_detectadas else "Ninguna"
        })

    except Exception as e:
        st.error(f"❌ Error al procesar el mensaje: {str(e)}")

# Mostrar gráfico de estadísticas
st.markdown("### 📊 Estadísticas de la sesión")
fig, ax = plt.subplots()
ax.bar(["SPAM", "NO SPAM"], [st.session_state.contador_spam, st.session_state.contador_nospam], color=["red", "green"])
ax.set_ylabel("Cantidad")
st.pyplot(fig)

import streamlit as st
import pandas as pd
from io import BytesIO

if st.session_state.historial:
    st.markdown("### 📝 Historial reciente")
    df_historial = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_historial.tail(3))

    # ----- crear Excel en memoria -----
    buffer = BytesIO()
    df_historial.to_excel(buffer, index=False, engine="openpyxl")  # escribe en el búfer
    buffer.seek(0)                                                # vuelve al inicio

    # ----- botón de descarga -----
    st.download_button(
        "⬇️ Descargar historial como Excel",
        data=buffer,
        file_name="historial_spam.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    # Sección de recursos adicionales
st.markdown("---")
st.markdown("## 📚 Recursos para aprender más")

st.markdown("""
Si quieres profundizar en los algoritmos utilizados para detección de spam y aprendizaje automático, aquí tienes algunos recursos útiles:

- [🔗 Curso de Machine Learning de Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)
- [🔗 Scikit-learn: Documentación oficial](https://scikit-learn.org/stable/documentation.html)
- [🔗 Introducción a Naive Bayes - Towards Data Science](https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0)
- [🔗 Clasificación de texto con Python (DataCamp)](https://www.datacamp.com/tutorial/text-analytics-python)
- [🔗 Libro gratuito: Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://github.com/ageron/handson-ml2)

> Estos recursos te ayudarán a entender cómo funcionan modelos como Naive Bayes, SVM, redes neuronales y más, aplicados al filtrado de spam.
""")
