import streamlit as st
import torch
from PIL import Image
import os
import time
from predict_module_grading import load_grading_model, load_stage_model, get_grading_prediction

# --- Page Configuration ---
st.set_page_config(page_title="Embryo AI Grade Pro", page_icon="🔬", layout="wide")

# --- Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;900&display=swap');
    .stApp { background: linear-gradient(180deg, #020617 0%, #0f172a 100%); color: #f8fafc; font-family: 'Outfit', sans-serif; }
    .main-title { font-size: 3.5rem; font-weight: 900; background: linear-gradient(to right, #00d4ff, #4f46e5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .sub-title { text-align: center; color: #64748b; font-size: 1.1rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 2rem; }
    .scan-container { position: relative; overflow: hidden; border-radius: 20px; border: 2px solid rgba(0, 212, 255, 0.2); }
    .scanning-line { position: absolute; width: 100%; height: 4px; background: #00d4ff; box-shadow: 0 0 15px #00d4ff; top: 0; animation: scan 3s linear infinite; z-index: 10; }
    @keyframes scan { 0% { top: 0; } 50% { top: 100%; } 100% { top: 0; } }
    .clinical-card { background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 24px; padding: 2rem; backdrop-filter: blur(20px); text-align: center; }
    .grade-hero { font-size: 7rem; font-weight: 900; color: #fff; text-shadow: 0 0 40px rgba(0, 212, 255, 0.6); line-height: 1; }
    .metric-box { background: rgba(15, 23, 42, 0.6); padding: 1.5rem; border-radius: 16px; border-top: 3px solid #00d4ff; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- Models ---
@st.cache_resource
def get_models(): return load_grading_model(), load_stage_model()
grading_model, stage_model = get_models()

# --- Header ---
st.markdown('<div class="main-title">EMBRYO AI PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Standard Clinical Diagnosis Suite</div>', unsafe_allow_html=True)

col_u, col_r = st.columns([1, 1.2], gap="large")

with col_u:
    uploaded_file = st.file_uploader("Upload Embryo Micrograph...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="scan-container"><div class="scanning-line"></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("⚡ ANALYZE NOW", use_container_width=True):
            with st.spinner("Analyzing Morphology..."):
                time.sleep(1.5)
                st.session_state['res_vfinal'] = get_grading_prediction(image, grading_model, stage_model)

with col_r:
    if 'res_vfinal' in st.session_state:
        res = st.session_state['res_vfinal']
        st.markdown("#### ANALYSIS RESULT")
        
        if res['type'] == 'gardner':
            st.markdown(f'''<div class="clinical-card">
                <div style="color:#94a3b8; letter-spacing:5px;">GARDNER SCORE</div>
                <div class="grade-hero">{res['final_score']}</div>
                <div style="color:#00d4ff; font-weight:700;">PHASE: BLASTOCYST (DAY 5)</div>
            </div>''', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f'<div class="metric-box"><div style="font-size:0.7rem; color:#64748b;">EXPANSION</div><div style="font-size:1.8rem; font-weight:800;">{res["expansion"]}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-box"><div style="font-size:0.7rem; color:#64748b;">ICM</div><div style="font-size:1.8rem; font-weight:800;">{res["icm"]}</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="metric-box"><div style="font-size:0.7rem; color:#64748b;">TE</div><div style="font-size:1.8rem; font-weight:800;">{res["te"]}</div></div>', unsafe_allow_html=True)
        
        elif res['type'] == 'cleavage':
            st.markdown(f'''<div class="clinical-card" style="border-color:#f59e0b;">
                <div style="color:#94a3b8; letter-spacing:5px;">CLEAVAGE GRADE</div>
                <div class="grade-hero" style="color:#f59e0b; text-shadow:0 0 40px rgba(245,158,11,0.4);">{res['final_score']}</div>
                <div style="color:#f59e0b; font-weight:700;">PHASE: CLEAVAGE ({res['stage'].upper()})</div>
                <hr style="opacity:0.1; margin:20px 0;">
                <div style="color:#94a3b8; font-size:0.9rem;">Fragmentation Audit: {res['description']}</div>
            </div>''', unsafe_allow_html=True)
            st.info("💡 Pro Tip: Grade 1 is the highest clinical quality for early-stage embryos.")
    else:
        st.info("Awaiting acquisition data...")

st.markdown("---")
st.sidebar.image("embryo_ai_logo_1776785134392.png", width=120)
st.sidebar.markdown("### Protocol V2.1")
st.sidebar.caption("SOTA: Multi-Standard Integration")
