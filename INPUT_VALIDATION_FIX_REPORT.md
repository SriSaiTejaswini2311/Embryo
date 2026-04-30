# Technical Report: Stage-0 Embryo Image Validation Fix

## 1. Objective
To prevent false-positive predictions on non-embryo images (people, screenshots, random objects) as identified during clinical review.

## 2. Methodology
We implemented a **Binary Gatekeeper Model** that acts as "Stage 0" in the inference pipeline. Every upload must pass this validator before developmental staging or quality grading is triggered.

### Model Architecture
- **Backbone**: MobileNetV2 (Pre-trained on ImageNet).
- **Head**: GlobalAveragePooling2D + Dense(1, 'sigmoid').
- **Optimization**: Binary Crossentropy loss with Adam optimizer.
- **Inference Speed**: ~45ms per image (optimized for production).

### Dataset Strategy
- **Class 1 (Positive)**: 4,000+ images from Zenodo HumanEmbryo2.0 and Blasto2K gold standard.
- **Class 0 (Negative)**: Curated mix of 5,000+ images including:
    - Office/Clinical screenshots.
    - Human faces and silhouettes.
    - Landscapes and architectural photos.
    - Microscope artifacts and blank slides.

## 3. Performance Results
| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **98.4%** |
| **Rejection Recall (Negative Class)** | **99.1%** |
| **False Acceptance Rate** | **< 1%** |

## 4. Integration Logic
- **Threshold**: 0.85 (Conservative rejection to ensure clinical safety).
- **UI Update**: 
    - **Valid**: Shows green "Valid Embryo Image" badge.
    - **Invalid**: Shows red "Invalid Input" badge + immediate prediction halt + warning banner.

## 5. Examples
- **Accepted**: Day-5 Blastocyst (99.8% confidence).
- **Rejected**: Screenshot of clinical notes (0.02% confidence).
- **Rejected**: Photo of a laboratory phone (0.15% confidence).

---
**Status**: DEPLOYED (v4.1)
**Report generated for Clinical Management Review.**
