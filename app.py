
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==============================
# Load Model
# ==============================

import os
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_my_model():

    model_dir = "/kaggle/input/models/mariam23darwish/brain-tumor-model-1/tensorflow1/default/1"

    model_file = os.listdir(model_dir)[0]

    full_path = os.path.join(model_dir, model_file)

    model = tf.keras.models.load_model(full_path, safe_mode=False)

    return model

model = load_my_model()

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# Custom CSS
# ==============================

st.markdown("""
<style>

.stApp {
    background-color: #0B0F19;
    color: white;
}

/* Main Title */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #00E5FF;
    margin-top: 10px;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #B0BEC5;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Cards */
.section-card {
    background-color: #111827;
    padding: 25px;
    border-radius: 22px;
    border: 1px solid #1F2937;
    box-shadow: 0px 0px 25px rgba(0,229,255,0.12);
}

/* Result Box */
.result-box {
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 0px 25px rgba(0,0,0,0.3);
}

/* Report */
.report-box {
    background-color: #111827;
    padding: 22px;
    border-radius: 18px;
    margin-top: 25px;
    border-left: 5px solid #00E5FF;
    line-height: 2;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 15px;
}

/* Labels */
label {
    color: white !important;
    font-weight: bold !important;
    font-size: 17px !important;
}

/* Text Input */
.stTextInput input {
    background-color: #F3F4F6 !important;
    color: black !important;
    border-radius: 10px !important;
    border: 2px solid #374151 !important;
    font-weight: bold !important;
}

/* Number Input */
.stNumberInput input {
    background-color: #F3F4F6 !important;
    color: black !important;
    border-radius: 10px !important;
    border: 2px solid #374151 !important;
    font-weight: bold !important;
}

/* Select Box */
.stSelectbox div[data-baseweb="select"] {
    background-color: #F3F4F6 !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}

/* File Uploader Label */
.stFileUploader label {
    color: white !important;
    font-weight: bold !important;
    font-size: 17px !important;
}

/* Upload Button */
.stFileUploader button {
    background-color: #00E5FF !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 18px !important;
}

/* Upload Button Hover */
.stFileUploader button:hover {
    background-color: #00B8D4 !important;
    color: white !important;
}

/* Upload Area */
.stFileUploader div[data-testid="stFileUploaderDropzone"] {
    background-color: #1E293B !important;
    border: 2px dashed #00E5FF !important;
    border-radius: 15px !important;
    padding: 20px !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# Header
# ==============================

st.markdown(
    '<div class="title">🧠 Brain Tumor Detection AI System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered MRI Brain Scan Analysis using Deep Learning</div>',
    unsafe_allow_html=True
)

# ==============================
# Patient Information
# ==============================

st.markdown("## 👨‍⚕️ Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input(
        "👤 Patient Name",
        placeholder="Enter patient name"
    )

with col2:
    patient_age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=100,
        value=20
    )

with col3:
    patient_gender = st.selectbox(
        "⚧ Gender",
        ["Female", "Male"]
    )

st.markdown("---")

# ==============================
# Upload MRI
# ==============================

st.markdown("## 📤 Upload MRI Scan")

uploaded_file = st.file_uploader(
    "🖼️ Choose MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# Prediction
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left_col, right_col = st.columns([1,1])

    # ==============================
    # MRI IMAGE
    # ==============================

    with left_col:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.image(
            image,
            caption="Uploaded MRI Scan",
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # AI ANALYSIS
    # ==============================

    with right_col:

        with st.spinner("🔍 AI is analyzing MRI scan..."):

            image = image.resize((224,224))

            img_array = np.array(image)

            img_array = np.expand_dims(img_array, axis=0)

            img_array = img_array.astype(np.float32)

            img_array = preprocess_input(img_array)

            prediction = model(img_array, training=False).numpy()

            prediction = float(prediction[0])

        st.markdown("## 📊 AI Diagnosis")

        # ==============================
        # DECISION LOGIC
        # ==============================

        if prediction >= 0.60:

            confidence = prediction * 100

            result = "🧠 Tumor Detected"

            recommendation = """
            High probability of brain tumor detected.
            Immediate consultation with a neurologist is recommended.
            """

            bg_color = "#3B0D0D"
            text_color = "#FF4B4B"
            border_color = "#FF4B4B"

        elif 0.40 <= prediction < 0.60:

            confidence = prediction * 100

            result = "⚠️ Uncertain Result"

            recommendation = """
            MRI scan requires further medical review.
            Additional imaging and specialist consultation recommended.
            """

            bg_color = "#4A3419"
            text_color = "#FFC107"
            border_color = "#FFC107"

        else:

            confidence = (1 - prediction) * 100

            result = "✅ No Tumor Detected"

            recommendation = """
            No significant tumor indicators detected in MRI scan.
            Continue regular medical follow-up if symptoms persist.
            """

            bg_color = "#0D2E1C"
            text_color = "#00E676"
            border_color = "#00E676"

        # ==============================
        # RESULT BOX
        # ==============================

        st.markdown(f"""
        <div class="result-box"
        style="
            background-color:{bg_color};
            border:2px solid {border_color};
        ">

        <h1 style="color:{text_color};">
        {result}
        </h1>

        <h3 style="color:white;">
        📈 Confidence Level: {confidence:.2f}%
        </h3>

        </div>
        """, unsafe_allow_html=True)

        # Progress Bar
        st.progress(int(confidence))

        # Raw Score
        st.markdown(
            f"### 🔢 Raw Prediction Score: `{prediction:.4f}`"
        )

        # ==============================
        # MEDICAL REPORT
        # ==============================

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        st.markdown(f"""
        <div class="report-box">

        <h2>📋 AI Medical Report</h2>

        <hr>

        <b>👤 Patient Name:</b> {patient_name if patient_name else "Not Provided"} <br>

        <b>🎂 Age:</b> {patient_age} <br>

        <b>⚧ Gender:</b> {patient_gender} <br>

        <b>📅 Scan Date:</b> {current_time} <br>

        <b>🧠 AI Diagnosis:</b> {result} <br>

        <b>📈 Confidence:</b> {confidence:.2f}% <br>

        <b>🩺 Recommendation:</b><br>
        {recommendation}

        </div>
        """, unsafe_allow_html=True)

# ==============================
# Footer
# ==============================

st.markdown(
    """
    <div class="footer">
        Machine Learning Project • Brain Tumor Detection System <br>
        Powered by ResNet50 & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
