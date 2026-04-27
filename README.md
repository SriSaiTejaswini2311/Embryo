# Embryo AI Analysis Suite

A multi-model deep learning system for automated human embryo developmental staging and Gardner quality grading.

## 🚀 Features
- **Developmental Staging**: Classifies embryos into 5 stages (2-cell, 4-cell, 8-cell, Morula, Blastocyst) using **MobileNetV2**.
- **Clinical Grading**: Expert Gardner grading (Expansion, ICM, TE) for blastocysts using **EfficientNet-B0** with Head-Weighted Hybrid Loss.
- **Explainability**: Integrated Grad-CAM heatmaps to visualize AI focus areas.
- **Clinical Watchdog**: High-threshold confidence auditing (70%) to flag uncertain predictions.

## 🛠️ Tech Stack
- **Framework**: Streamlit (Dashboard)
- **Deep Learning**: PyTorch (Grading) & TensorFlow (Staging)
- **Deployment**: Streamlit Community Cloud / Vercel

## 📦 Local Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd embryo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app_unified.py
   ```

## 🔬 Model Technicals
- **Staging Model**: Sequential CNN based on MobileNetV2. Optimized for speed/accuracy balance.
- **Grading Model (v4.0)**: Supervised Multi-Head EfficientNet-B0. Trained on the Blasto2K gold-standard dataset.
- **Confidence Logic**: Weighted mean of multi-head softmax outputs.

## ⚠️ Disclaimer
This tool is for research and clinical auditing purposes only. AI predictions should always be reviewed by a certified embryologist before any clinical decisions are made.

---
**Developed for: Malpani Infertility Clinic**
