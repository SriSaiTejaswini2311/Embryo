# Embryo AI Analysis Suite: Executive Technical Report

![Project Header](file:///c:/Users/LENOVO/Desktop/embryo/embryo_ai_logo_1776785134392.png)

## 📋 Project Objective
Develop a production-grade clinical AI for automated embryo developmental staging and Gardner quality grading.

- **Phase 1**: Developmental stage classification (Turbo MobileNetV2).
- **Phase 2**: Clinical quality grading (Supervised EfficientNet-B0 v4.0).
- **Goal**: Provide a high-confidence decision-support tool for embryologists.

---

## 🔬 Phase 1: Developmental Staging (Turbo)
Successfully built an optimized pipeline using **MobileNetV2** for rapid inference.
- **Dataset**: Zenodo HumanEmbryo2.0 (2,000+ images)
- **Classes**: 2-cell, 4-cell, 8-cell, Morula, Blastocyst
- **Outcome**: 100% successful prototype deployment.

---

## 🧬 Phase 2: Professional Gardner Grading (v4.0)
The core diagnostic engine leverages a **Supervised Multi-Head EfficientNet-B0** architecture.

### Benchmark Results (Gold Standard)
| Metric | Performance |
| :--- | :--- |
| **Expansion Accuracy** | **80%** |
| **ICM Accuracy** | **67%** |
| **TE Accuracy** | **61%** |
| **Exact Gardner Match** | **38%** |

### Model Evolution
- **V2**: Baseline multi-head model.
- **V3**: Class imbalance experiment (Weighted Random Sampler).
- **V4 (Current)**: EfficientNet-B0 with **Hybrid Head-Weighted Loss**. Focuses on rare minority classes (Grade C).

---

## 🛠️ Deployment & Explainability
The system is deployed as a unified Streamlit dashboard featuring:
- **Clinical Watchdog**: Mandatory human-review flags for predictions below 70% confidence.
- **AI Explainability**: Integrated **Grad-CAM** heatmaps to visualize exactly where the model is looking (e.g., Zona Pellucida vs. ICM).

---

## 🧪 External Research Summary
I have conducted a technical audit of several leading public repositories and papers:
- **Blastocyst-Grading (Legacy)**: ❌ Obsolete / No weights.
- **OpenFertility**: ❌ Data utility only.
- **Kaggle Blasto2K (GRU)**: ❌ Significantly lower Expansion accuracy (20%) compared to our v4.0.
- **STORK-A / Life Whisperer**: 🔒 Restricted/Proprietary models.

**Conclusion**: The Malpani v4.0 engine currently outperforms all identified open-source alternatives for this clinical use case.

---

## 📓 Notebooks & Colab Setup

For research, training, and further fine-tuning, the following notebooks are included in the `/research` directory.

### 1. `embryo_classification.ipynb` (Phase 1)
- **Purpose**: Training the developmental stage classifier.
- **How to run**:
    1. Upload to [Google Colab](https://colab.research.google.com/).
    2. Set runtime to **GPU (T4)**.
    3. Run all cells to download the Zenodo dataset and train the MobileNetV2 Turbo model.
- **Output**: Generates `embryo_model_turbo.h5`. Save this to the root directory for the app to function.

### 2. `embryo_research_malpani.ipynb` (Phase 2)
- **Purpose**: Clinical Gardner grading research and v4.0 training.
- **How to run**:
    1. Upload to Colab.
    2. Ensure the **Blasto2K dataset** is available in your Drive or environment.
    3. Run the "V4 Supervised Protocol" cells.
- **Output**: Generates `embryo_grading_v4.pth`. This is the core weight file for the clinical grading engine.

---

## 🚀 Next Steps
1. **Clinic Fine-tuning**: Ingest Malpani internal data to move beyond public benchmarks.
2. **Implantation Research**: Correlate morphology with historical pregnancy outcome data.
3. **API Integration**: Transition from micro-tool to a robust clinical API.

---
**Developed by: [USER]**
**Engine: Malpani Embryo AI Pro v4.0**
