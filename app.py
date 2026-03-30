import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import math

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Recomendador de Gafas IA", layout="centered")

st.title("👓 Recomendador de Gafas con IA")
st.write("Sube una foto y descubre qué gafas te quedan mejor")

# ---------------- FUNCIONES ----------------

def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def clasificar_rostro(ratio_alto_ancho, ratio_mandibula_pomulos,
                      ratio_frente_mandibula, ratio_frente_pomulos):

    if ratio_alto_ancho >= 1.36 and ratio_mandibula_pomulos >= 0.75:
        return "rectangular"
    if ratio_alto_ancho < 1.30 and ratio_mandibula_pomulos >= 0.82:
        return "cuadrado"
    if ratio_frente_mandibula <= 0.90:
        return "triangular"
    if ratio_frente_pomulos <= 0.85 and ratio_mandibula_pomulos < 0.75:
        return "diamante"
    if 1.25 <= ratio_alto_ancho < 1.36:
        return "ovalado"
    return "redondo"

RECOMENDACIONES = {
    "ovalado": ["aviador", "cuadradas", "cat-eye"],
    "redondo": ["rectangulares", "cuadradas", "cat-eye"],
    "cuadrado": ["redondas", "ovaladas", "aviador"],
    "rectangular": ["oversized", "grandes", "anchas"],
    "diamante": ["cat-eye", "ovaladas", "suaves"],
    "triangular": ["cat-eye", "aviador", "superior fuerte"],
}

# ---------------- MODELO ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# ---------------- INTERFAZ ----------------
uploaded_file = st.file_uploader("📤 Sube tu foto", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Convertir imagen
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        st.error("❌ No se detectó rostro")
    else:
        h, w, _ = img.shape
        lm = results.multi_face_landmarks[0].landmark

        def p(i):
            return (int(lm[i].x * w), int(lm[i].y * h))

        # puntos clave
        frente = p(10)
        barbilla = p(152)
        lado_izq = p(234)
        lado_der = p(454)
        pomulo_izq = p(93)
        pomulo_der = p(323)
        mandibula_izq = p(172)
        mandibula_der = p(397)
        frente_izq = p(54)
        frente_der = p(284)

        # medidas
        alto = distancia(frente, barbilla)
        ancho = distancia(lado_izq, lado_der)
        pomulos = distancia(pomulo_izq, pomulo_der)
        mandibula = distancia(mandibula_izq, mandibula_der)
        frente_ancho = distancia(frente_izq, frente_der)

        # ratios
        r1 = alto / ancho
        r2 = mandibula / pomulos
        r3 = frente_ancho / mandibula
        r4 = frente_ancho / pomulos

        tipo = clasificar_rostro(r1, r2, r3, r4)

        st.success(f"🧠 Tipo de rostro: {tipo.upper()}")

        # recomendaciones
        st.subheader("👓 Gafas recomendadas:")
        for g in RECOMENDACIONES[tipo]:
            st.write(f"✔ {g}")

        st.info("💡 Próximamente: simulación con IA")
