import streamlit as st
import tensorflow as tf
import keras  # Keras 3
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# PAGE CONFIG (UI UPGRADE)
# =========================
st.set_page_config(page_title="Insect AI", layout="centered")

st.title("🦟 Insect Classification AI")
st.caption("Powered by Deep Learning (MobileNetV2)")
st.divider()

# =========================
# DOWNLOAD MODEL FROM DRIVE
# =========================

FILE_ID = "1aVcHWqBMBukgRfWIcVU0UdzY3djSxQ5o"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "insect_model.keras"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_trained_model()

# 🔥 Handle model load failure safely
if model is None:
    st.stop()

# =========================
# CLASS NAMES
# =========================

class_names = [
    "Aedes_aegypti","Aedes_albopictus","Aedes_vexans",
    "Amblyomma_americanum","Anopheles_stephensi",
    "Anopheles_tessellatus","Cimex_lectularius",
    "Ctenocephalides_canis","Ctenocephalides_felis",
    "Culex_quinquefasciatus","Culex_vishnui",
    "Ixodes_ricinus","Ixodes_scapularis",
    "Pediculus_humanus_capitis",
    "Pediculus_humanus_corporis",
    "Rhipicephalus_sanguineus"
]

# =========================
# FILE UPLOAD UI
# =========================

uploaded_file = st.file_uploader(
    "📤 Upload an insect image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION PIPELINE
# =========================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.divider()

    # Preprocess
    img = image.resize((192, 192))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Classifying..."):
        prediction = model.predict(img_array)[0]

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # =========================
    # OUTPUT (ENHANCED UI)
    # =========================

    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}")

    # 🔥 Confidence bar
    st.progress(confidence)

    st.subheader("🔝 Top 3 Predictions")

    top_indices = np.argsort(prediction)[-3:][::-1]

    for i in top_indices:
        st.write(f"{class_names[i]} — {prediction[i]:.2f}")
