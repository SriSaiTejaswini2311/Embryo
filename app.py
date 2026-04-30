import streamlit as st
import os
from predict_module import EmbryoClassifier
from PIL import Image
import numpy as np
import cv2

# Page Config
st.set_page_config(
    page_title="Embryo AI - Stage Classifier",
    page_icon="🔬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc99;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e2130;
        border: 1px solid #3e4150;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Embryo Developmental Stage Classifier")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Path", "embryo_model_turbo.h5")

@st.cache_resource
def load_classifier(path):
    return EmbryoClassifier(path)

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please train the model first using the Colab notebook.")
    st.stop()

with st.spinner("Loading AI Model weights (this may take a minute on first run)..."):
    classifier = load_classifier(model_path)

# Main UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Embryo Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save temp file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

if uploaded_file is not None:
    with col2:
        st.subheader("AI Analysis Results")
        
        with st.spinner("Analyzing developmental stage..."):
            pred_class, confidence, all_probs = classifier.predict("temp_image.jpg")
            
            # Display Prediction
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style='color: #00cc99; text-align: center;'>{pred_class}</h2>
                <h4 style='text-align: center;'>Confidence: {confidence*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution
            st.markdown("### Confidence Distribution")
            for i, (cls, prob) in enumerate(zip(classifier.classes, all_probs)):
                st.write(f"{cls}")
                st.progress(float(prob))

            # Grad-CAM Visualization
            st.markdown("---")
            st.subheader("Focus Region (Grad-CAM)")
            gradcam_img = classifier.get_gradcam("temp_image.jpg")
            # Convert BGR (OpenCV) to RGB for Streamlit
            gradcam_img_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
            st.image(gradcam_img_rgb, caption="Heatmap showing areas of neural focus", use_column_width=True)

st.markdown("---")
st.caption("Embryo AI Phase 1 Prototype | Powered by EfficientNetB0")
