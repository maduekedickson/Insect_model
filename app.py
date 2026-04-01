import streamlit as st
import tensorflow as tf
import keras  # Use native keras 3
import numpy as np
from PIL import Image
import gdown
import os

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
        # Use native keras to load the Keras 3 model
        # We use compile=False to avoid any optimizer-related dependency issues
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

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
# UI
# =========================

st.title("🦟 Insect Classification App")
st.write("Upload an insect image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((192,192))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Classifying..."):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")
