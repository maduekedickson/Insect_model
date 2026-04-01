import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# =========================
# DOWNLOAD MODEL FROM DRIVE
# =========================

MODEL_URL = "https://drive.google.com/file/d/1aVcHWqBMBukgRfWIcVU0UdzY3djSxQ5o/view?usp=sharing"
MODEL_PATH = "model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# CLASS NAMES (IMPORTANT)
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
# UI
# =========================

st.title("🦟 Insect Classification App")
st.write("Upload an insect image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((192,192))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")