import streamlit as st
import os
import time
from PIL import Image
import numpy as np

# Import our custom predictors
from predict_module import EmbryoClassifier
from predict_malpani import EmbryoPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Embryo AI - Clinical Diagnostic Suite",
    page_icon="🔬",
    layout="wide"
)

# --- Custom Styling (Premium Dark Mode) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020c1b);
        color: #e6f1ff;
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(10, 25, 47, 0.8);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(100, 255, 218, 0.1);
        margin-bottom: 3rem;
        border-radius: 0 0 30px 30px;
    }
    
    .title-gradient {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #64ffda 0%, #48bfe3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .subtitle {
        color: #8892b0;
        font-size: 0.9rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 5px;
    }

    .report-container {
        background: rgba(17, 34, 64, 0.4);
        border: 1px solid rgba(100, 255, 218, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .metric-card {
        background: rgba(2, 12, 27, 0.6);
        border: 1px solid rgba(100, 255, 218, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #64ffda;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 10px;
    }
    
    .conf-bar-bg {
        height: 8px;
        background: #112240;
        border-radius: 4px;
        margin-top: 12px;
        overflow: hidden;
    }
    
    .conf-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #64ffda, #48bfe3);
        border-radius: 4px;
    }
    
    .warning-banner {
        background: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
        color: #ff4b4b;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('''
<div class="main-header">
    <div class="title-gradient">EMBRYO AI ANALYTICS</div>
    <div class="subtitle">Multi-Model Clinical Diagnostic Suite</div>
</div>
''', unsafe_allow_html=True)

# --- Session State ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- Layout ---
col_up, col_res = st.columns([1, 2], gap="large")

with col_up:
    st.markdown("### 📤 Image Ingestion")
    uploaded_file = st.file_uploader("Upload Embryo Micrograph", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Patient Sample", use_container_width=True)
        
        if st.button("🚀 RUN FULL DIAGNOSTIC SCAN", use_container_width=True):
            with st.spinner("Executing Multi-Model Audit..."):
                # Save temp file
                temp_path = "current_sample.png"
                img.save(temp_path)
                
                # Model 1: Stage Classification (MobileNetV2 Turbo)
                classifier = EmbryoClassifier(model_path="embryo_model_turbo.h5")
                stage, stage_conf, _ = classifier.predict(temp_path)
                
                # Model 2: Quality Grading (EfficientNet-B0 Supervised)
                # Only if it's a Blastocyst
                grading_res = None
                if stage == "Blastocyst":
                    grader = EmbryoPredictor(model_path="embryo_grading_v4.pth")
                    full_grade, grading_res = grader.predict(temp_path)
                
                st.session_state.results = {
                    "stage": stage,
                    "stage_conf": stage_conf,
                    "grading": grading_res,
                    "img_path": temp_path
                }
                st.session_state.analysis_done = True

with col_res:
    if st.session_state.analysis_done:
        res = st.session_state.results
        
        # 1. Warning System
        if res['grading'] and res['grading']['low_confidence']:
            st.markdown('''
            <div class="warning-banner">
                ⚠️ CLINICAL ALERT: Low confidence detection. Prediction requires manual expert verification.
            </div>
            ''', unsafe_allow_html=True)
            
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        
        st.markdown("### 🔬 Automated Diagnostic Report")
        st.divider()
        
        # Primary Metrics
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{res['stage']}</div>
                <div class="metric-label">Developmental Stage</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with m2:
            grade_val = res['grading']['full_grade'] if res['grading'] else "N/A"
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{grade_val}</div>
                <div class="metric-label">Gardner Grade</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with m3:
            conf = res['grading']['confidence'] if res['grading'] else res['stage_conf']
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{conf:.1%}</div>
                <div class="metric-label">Aggregate Confidence</div>
            </div>
            ''', unsafe_allow_html=True)
            
        st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
        
        # Detailed Audit
        if res['grading']:
            st.markdown("#### Clinical Morphology Audit")
            det = res['grading']
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.write(f"**Expansion**: {det['expansion']}")
                st.markdown(f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{det["head_confidences"]["expansion"]*100}%"></div></div>', unsafe_allow_html=True)
            with c2:
                st.write(f"**ICM**: {det['icm']}")
                st.markdown(f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{det["head_confidences"]["icm"]*100}%"></div></div>', unsafe_allow_html=True)
            with c3:
                st.write(f"**TE**: {det['te']}")
                st.markdown(f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{det["head_confidences"]["te"]*100}%"></div></div>', unsafe_allow_html=True)
        else:
            st.info(f"💡 Quality grading is only applicable for Blastocyst stage. Current sample identified as {res['stage']}.")
            
        st.divider()
        st.caption("🛡️ Model Suite: MobileNetV2 (Staging) | EfficientNet-B0 (Grading v4.0)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Explainability (Grad-CAM)
        if st.checkbox("Show AI Focus (Explainability Map)"):
            classifier = EmbryoClassifier(model_path="embryo_model_turbo.h5")
            gcam = classifier.get_gradcam(res['img_path'])
            st.image(gcam, caption="Grad-CAM Heatmap (Focus Areas)", use_container_width=True)
            
    else:
        st.markdown('''
        <div style="text-align:center; padding: 5rem; border: 2px dashed rgba(100, 255, 218, 0.1); border-radius: 24px;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
            <div style="color: #8892b0;">Waiting for image upload to begin analysis...</div>
        </div>
        ''', unsafe_allow_html=True)
