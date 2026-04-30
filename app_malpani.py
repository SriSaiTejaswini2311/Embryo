import streamlit as st
import os
import time
from PIL import Image
from predict_malpani import EmbryoPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Embryo AI - Malpani Pro v4.0",
    page_icon="🔬",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background: linear-gradient(135deg, #05161a 0%, #072e33 100%); color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .main-header { text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(0, 212, 255, 0.1); margin-bottom: 2rem; }
    .malpani-logo { font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; background: linear-gradient(to right, #00d4ff, #2997ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .report-card { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 2rem; }
    .conf-bar { height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; margin-top: 5px; }
    .conf-fill { height: 100%; background: #38bdf8; }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="main-header">
    <div class="malpani-logo">EMBRYO AI PRO v4.0</div>
    <div style="color:#64748b; font-size:0.8rem; letter-spacing:2px; margin-top:5px;">EFFICIENTNET-B0 HYBRID AUDIT</div>
</div>
''', unsafe_allow_html=True)

col_sidebar, col_main = st.columns([1, 2.5], gap="large")

with col_sidebar:
    st.markdown("### 🎚️ Audit Protocol")
    uploaded_file = st.file_uploader("Select Cell Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img_view = Image.open(uploaded_file)
        st.image(img_view, use_container_width=True)
        if st.button("⚡ EXECUTE V4.0 SCAN", use_container_width=True):
            with st.spinner("Analyzing Morphology..."):
                temp_path = "temp_v4_eval.png"
                img_view.save(temp_path)
                
                # Run V4 Supervised Predictor
                engine = EmbryoPredictor(model_path="embryo_grading_v4.pth")
                grade, details = engine.predict(temp_path)
                
                st.session_state['v4_result'] = {"grade": grade, "details": details}

with col_main:
    if 'v4_result' in st.session_state:
        res = st.session_state['v4_result']
        det = res['details']
        
        st.markdown('### 🧪 Diagnostic Audit: Day 5/6')
        
        # Low Confidence Warning
        if det['low_confidence']:
            st.warning("⚠️ **Low Confidence Prediction**: The AI is uncertain about this morphology. Clinical human review is strictly required.")
            
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown(f'<div style="color:#94a3b8; font-size:0.85rem;">FINAL DIAGNOSIS</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:3.5rem; font-weight:800; color:#fff; line-height:1;">{res["grade"]}</div>', unsafe_allow_html=True)
            st.write(f"**Expansion Stage**: {det['expansion']}")
            st.write(f"**ICM Integrity**: {det['icm']} | **TE Integrity**: {det['te']}")
        with c2:
            st.markdown("#### Clinical Confidence Audit")
            
            # Per-Head Confidence Display
            h_conf = det['head_confidences']
            for head, val in [("Expansion", h_conf['expansion']), ("ICM", h_conf['icm']), ("TE", h_conf['te'])]:
                st.write(f"{head}: {val:.1%}")
                st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="width:{val*100}%"></div></div>', unsafe_allow_html=True)
                st.markdown('<div style="margin-bottom:10px;"></div>', unsafe_allow_html=True)
            
            avg_conf = det['confidence']
            st.write(f"**Overall Mean Confidence: {avg_conf:.1%}**")
            
        st.divider()
        st.caption("🛡️ Engine: EfficientNet-B0 Supervised | Head-Weighted Hybrid Loss v4.0")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload sample to initialize Version 4.0 analysis.")
