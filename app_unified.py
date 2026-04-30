import streamlit as st
import os
from PIL import Image
import numpy as np

from predict_module import EmbryoClassifier
from predict_malpani import EmbryoPredictor

st.set_page_config(
    page_title="Embryo AI — Clinical Diagnostic Suite",
    page_icon="🔬",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────
#  GLOBAL CSS  (premium dark, custom buttons, every element)
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp {
    background: linear-gradient(135deg, #020c1b 0%, #0a192f 60%, #0d2137 100%);
    color: #ccd6f6;
    min-height: 100vh;
}
/* hide hamburger & footer */
#MainMenu, footer { visibility: hidden; }

/* ── Header ── */
.hero {
    text-align: center;
    padding: 3rem 2rem 2.5rem;
    background: linear-gradient(180deg, rgba(10,25,47,0.95) 0%, rgba(2,12,27,0) 100%);
    border-bottom: 1px solid rgba(100,255,218,0.07);
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #64ffda;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg, #e6f1ff 0%, #64ffda 60%, #48bfe3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-size: 0.88rem;
    color: #8892b0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ── Section headings ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #64ffda;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(100,255,218,0.12);
}

/* ── Sample thumbnails ── */
.sample-card {
    background: rgba(17,34,64,0.6);
    border: 1px solid rgba(100,255,218,0.08);
    border-radius: 14px;
    padding: 10px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.sample-card:hover { border-color: rgba(100,255,218,0.35); transform: translateY(-3px); }
.sample-label {
    font-size: 0.72rem;
    color: #8892b0;
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── ALL Streamlit buttons → teal pill style ── */
.stButton > button {
    background: linear-gradient(135deg, #0d7377 0%, #14a085 100%) !important;
    color: #e6f1ff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.45rem 0.8rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(13,115,119,0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #14a085 0%, #64ffda 100%) !important;
    color: #020c1b !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(100,255,218,0.35) !important;
}

/* ── Primary CTA button ── */
div[data-testid="stButton"] > button[kind="primary"],
.cta-btn > button {
    background: linear-gradient(135deg, #1de9b6 0%, #1976d2 100%) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 24px rgba(29,233,182,0.3) !important;
    letter-spacing: 1px !important;
}

/* ── Left panel ── */
.left-panel {
    background: rgba(10,25,47,0.55);
    border: 1px solid rgba(100,255,218,0.06);
    border-radius: 24px;
    padding: 1.6rem;
    backdrop-filter: blur(12px);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(17,34,64,0.5) !important;
    border: 1.5px dashed rgba(100,255,218,0.2) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(100,255,218,0.5) !important;
}

/* ── Report card ── */
.report-card {
    background: rgba(13,24,48,0.7);
    border: 1px solid rgba(100,255,218,0.1);
    border-radius: 24px;
    padding: 2.2rem;
    box-shadow: 0 24px 60px rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
}

/* ── Metric tiles ── */
.metric-tile {
    background: rgba(2,12,27,0.8);
    border: 1px solid rgba(100,255,218,0.08);
    border-radius: 18px;
    padding: 1.4rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-tile:hover { border-color: rgba(100,255,218,0.25); }
.metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: #64ffda;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-lbl {
    font-size: 0.65rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* ── Confidence bars ── */
.conf-row { margin-bottom: 14px; }
.conf-title { font-size: 0.78rem; color: #ccd6f6; font-weight: 500; margin-bottom: 5px; display:flex; justify-content:space-between; }
.conf-track { height: 7px; background: #112240; border-radius: 4px; overflow: hidden; }
.conf-fill { height: 100%; background: linear-gradient(90deg, #0d7377, #64ffda); border-radius: 4px; transition: width 0.6s ease; }

/* ── Warning banner ── */
.warn-box {
    background: rgba(255,85,85,0.08);
    border-left: 4px solid #ff5555;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.4rem;
    color: #ff8080;
    font-size: 0.85rem;
    font-weight: 600;
}

/* ── Gallery cards ── */
.gallery-card {
    background: rgba(10,25,47,0.6);
    border: 1px solid rgba(100,255,218,0.07);
    border-radius: 20px;
    padding: 1.2rem;
    transition: border-color 0.25s, transform 0.25s;
}
.gallery-card:hover { border-color: rgba(100,255,218,0.3); transform: translateY(-4px); }
.gallery-stage { font-size: 0.7rem; font-weight: 700; color: #64ffda; letter-spacing: 2px; text-transform: uppercase; margin-top: 10px; }
.gallery-grade { font-size: 1.5rem; font-weight: 800; color: #e6f1ff; margin: 4px 0; }
.gallery-explain { font-size: 0.78rem; color: #8892b0; line-height: 1.55; margin-top: 8px; }

/* ── Divider ── */
hr { border-color: rgba(100,255,218,0.06) !important; }

/* ── Streamlit image caption ── */
[data-testid="caption"] { color: #8892b0 !important; font-size: 0.72rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🔬 Powered by EfficientNet-B0 + MobileNetV2</div>
    <div class="hero-title">Embryo AI Analytics</div>
    <div class="hero-sub">Multi-Model Clinical Diagnostic Suite · Gardner Grading · Developmental Staging</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for key, val in [('analysis_done', False), ('trigger_scan', False), ('use_sample', None)]:
    if key not in st.session_state:
        st.session_state[key] = val

SAMPLES = {
    "btn1": ("sample_images/2_cell_sample.jpeg",   "Cleavage Stage"),
    "btn2": ("sample_images/morula_sample.jpeg",    "Morula Stage"),
    "btn3": ("sample_images/blastocyst_good.png",   "Blastocyst · Good"),
    "btn4": ("sample_images/blastocyst_poor.png",   "Blastocyst · Poor"),
}

# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 2.2], gap="large")

# ──────────────────────────────────────────────────────────────
#  LEFT PANEL
# ──────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    # Quick-Start Samples
    st.markdown('<div class="section-label">⚡ Quick Start · Try a Sample</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    for col_widget, (key, (path, label)) in zip([r1, r2, r3, r4], SAMPLES.items()):
        with col_widget:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            st.markdown(f'<div class="sample-label">{label.split("·")[0].strip()}</div>', unsafe_allow_html=True)
            if st.button("▶ Run", key=key, use_container_width=True):
                st.session_state.use_sample = path
                st.session_state.trigger_scan = True

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📤 Upload Your Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag & drop or browse", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    img_to_scan = None
    if uploaded_file:
        img_to_scan = Image.open(uploaded_file)
        st.session_state.use_sample = None
    elif st.session_state.use_sample and os.path.exists(st.session_state.use_sample):
        img_to_scan = Image.open(st.session_state.use_sample)

    if img_to_scan:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(img_to_scan, caption="Loaded Sample", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        run_clicked = st.button("🚀  RUN DIAGNOSTIC SCAN", use_container_width=True)

        if run_clicked or st.session_state.trigger_scan:
            st.session_state.trigger_scan = False
            with st.spinner("Running multi-model analysis…"):
                temp_path = "current_sample.png"
                img_to_scan.save(temp_path)

                classifier = EmbryoClassifier(model_path="embryo_model_turbo.h5")
                stage, stage_conf, _ = classifier.predict(temp_path)

                grading_res = None
                if stage == "Blastocyst":
                    grader = EmbryoPredictor(model_path="embryo_grading_v4.pth")
                    _, grading_res = grader.predict(temp_path)

                st.session_state.results = {
                    "stage": stage,
                    "stage_conf": stage_conf,
                    "grading": grading_res,
                    "img_path": temp_path,
                }
                st.session_state.analysis_done = True

    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  RIGHT PANEL — RESULTS
# ──────────────────────────────────────────────────────────────
with col_right:
    if st.session_state.analysis_done:
        res = st.session_state.results

        # ── Clinical alert ──
        if res["grading"] and res["grading"]["low_confidence"]:
            st.markdown("""
            <div class="warn-box">
                ⚠️ LOW CONFIDENCE — AI certainty is below clinical threshold (70%).
                Manual embryologist review is mandatory before any decision.
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🧬 Automated Diagnostic Report</div>', unsafe_allow_html=True)

        # ── Top metric tiles ──
        t1, t2, t3 = st.columns(3)
        grade_val = res["grading"]["full_grade"] if res["grading"] else "—"
        conf_val  = res["grading"]["confidence"] if res["grading"] else res["stage_conf"]

        with t1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-val">{res['stage'].replace('-', '‑')}</div>
                <div class="metric-lbl">Developmental Stage</div>
            </div>""", unsafe_allow_html=True)
        with t2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-val">{grade_val}</div>
                <div class="metric-lbl">Gardner Grade</div>
            </div>""", unsafe_allow_html=True)
        with t3:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-val">{conf_val:.0%}</div>
                <div class="metric-lbl">Aggregate Confidence</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Morphology audit (blastocyst only) ──
        if res["grading"]:
            st.markdown('<div class="section-label">🔍 Morphology Head Audit</div>', unsafe_allow_html=True)
            det = res["grading"]
            hc  = det["head_confidences"]
            for name, val in [("Expansion Stage", hc["expansion"]), ("ICM Integrity", hc["icm"]), ("TE Integrity", hc["te"])]:
                st.markdown(f"""
                <div class="conf-row">
                    <div class="conf-title"><span>{name}</span><span style="color:#64ffda;font-weight:700">{val:.0%}</span></div>
                    <div class="conf-track"><div class="conf-fill" style="width:{val*100:.1f}%"></div></div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <p style="font-size:0.8rem;color:#8892b0;margin-top:6px;">
                Expansion: <b style="color:#ccd6f6">{det['expansion']}</b> &nbsp;·&nbsp;
                ICM: <b style="color:#ccd6f6">{det['icm']}</b> &nbsp;·&nbsp;
                TE: <b style="color:#ccd6f6">{det['te']}</b>
            </p>""", unsafe_allow_html=True)
        else:
            st.info(f"ℹ️  Gardner grading applies to Blastocysts only. This sample was classified as **{res['stage']}**.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.72rem;color:#4a5568;">🛡️ Engine: MobileNetV2 Turbo (Staging) &nbsp;|&nbsp; EfficientNet-B0 Supervised (Grading v4.0) &nbsp;|&nbsp; Blasto2K Gold Standard Dataset</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Grad-CAM ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.checkbox("🔥  Show Grad-CAM Explainability Heatmap"):
            with st.spinner("Generating focus map…"):
                clf = EmbryoClassifier(model_path="embryo_model_turbo.h5")
                gcam = clf.get_gradcam(res["img_path"])
                st.image(gcam, caption="Grad-CAM · Red = AI focus regions", use_container_width=True)

    else:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            padding: 6rem 2rem;
            border: 1.5px dashed rgba(100,255,218,0.1);
            border-radius: 24px;
            background: rgba(10,25,47,0.3);
            text-align: center;
        ">
            <div style="font-size:3.5rem;margin-bottom:1.2rem;opacity:0.6">🔬</div>
            <div style="font-size:1.05rem;font-weight:600;color:#ccd6f6;margin-bottom:6px">
                No Sample Loaded
            </div>
            <div style="font-size:0.82rem;color:#4a5568;max-width:320px;line-height:1.6">
                Upload a microscope image or click one of the <strong style="color:#64ffda">Quick Start</strong> samples on the left to begin analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  REFERENCE GALLERY
# ══════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">📚 Reference Gallery · Expected Inputs & Outputs</div>', unsafe_allow_html=True)

gallery_data = [
    {
        "img": "sample_images/2_cell_sample.jpeg",
        "stage": "Cleavage Stage",
        "grade": "N/A",
        "explain": "Early 2–4 cell embryo. Gardner grading does not apply. The AI evaluates blastomere symmetry and fragmentation levels.",
    },
    {
        "img": "sample_images/morula_sample.jpeg",
        "stage": "Morula",
        "grade": "N/A",
        "explain": "Compact morula (Day 3–4). Cells are tightly compacted. Still pre-blastocyst — Gardner grading is not applicable here.",
    },
    {
        "img": "sample_images/blastocyst_good.png",
        "stage": "Blastocyst",
        "grade": "4AA",
        "explain": "Expanded blastocyst with a large, well-defined ICM and a tightly cohesive TE cell layer. Top-tier implantation candidate.",
    },
    {
        "img": "sample_images/blastocyst_poor.png",
        "stage": "Blastocyst",
        "grade": "3CC",
        "explain": "Partially expanded blastocyst. The ICM appears sparse and the TE cells are loose/irregular — resulting in 'C' quality grades.",
    },
]

g1, g2, g3, g4 = st.columns(4)
for col_widget, item in zip([g1, g2, g3, g4], gallery_data):
    with col_widget:
        st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
        if os.path.exists(item["img"]):
            st.image(item["img"], use_container_width=True)
        st.markdown(f'<div class="gallery-stage">{item["stage"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gallery-grade">{item["grade"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gallery-explain">{item["explain"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
