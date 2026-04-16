# Vehicle Damage Detection using Transfer Learning & Explainable AI

An end-to-end deep learning project for automated vehicle damage classification from images, built using TensorFlow/Keras and optimized for real-world insurance / fleet inspection workflows.

---

## Business Problem

Manual vehicle damage assessment is slow, costly, and difficult to scale across insurance, rental, and fleet management operations.

This project develops an AI-powered image classifier capable of automatically determining whether a vehicle image contains visible damage, enabling faster triage and automated inspection pipelines.

---

## Project Objectives

- Build a robust binary image classifier for:
  - **Damage**
  - **Whole / Undamaged**
- Optimize prediction thresholds based on business priorities rather than default model outputs
- Interpret model decisions using explainable AI (Grad-CAM)
- Demonstrate production-oriented ML thinking through evaluation and calibration

---

## Dataset Overview

- **Total Images:** 2,300
- **Classes:**  
  - Damage  
  - Whole / Undamaged  

### Data Split

| Split | Images |
|--------|--------|
| Train | 1,610 |
| Validation | 345 |
| Test | 345 |

- Stratified splitting applied
- Duplicate / collision checks performed during preprocessing
- Resolution and aspect ratio analysis conducted during EDA

---

## Modeling Pipeline

### 1. Data Preparation & EDA
- Image resolution analysis
- Aspect ratio distribution study
- Class balance verification
- Stratified train/val/test split generation

### 2. Transfer Learning Baseline
- Pretrained **MobileNetV2** backbone
- Frozen feature extractor training phase

### 3. Fine-Tuning
- Partial unfreezing of upper convolutional blocks
- Low learning-rate fine-tuning for representation adaptation

### 4. Threshold Optimization
- Swept classification thresholds from **0.10 → 0.90**
- Selected optimal threshold based on **Damage F1 Score**

### 5. Explainability
- Grad-CAM visualizations for model interpretability
- Failure-mode analysis on misclassifications

---

## Final Performance

### Optimized Threshold = **0.15**

| Metric | Score |
|--------|-------|
| Accuracy | 93.0% |
| Damage Precision | 92.5% |
| Damage Recall | 93.1% |
| Damage F1 | 92.8% |

---

## Threshold Optimization Impact

Default sigmoid threshold (0.50) underperformed for the business objective of maximizing damage detection.

### Performance Comparison

| Threshold | Damage Precision | Damage Recall | Damage F1 |
|----------|-----------------|--------------|----------|
| 0.50 | 97.5% | 68.2% | 80.3% |
| **0.15** | **92.5%** | **93.1%** | **92.8%** |

Threshold calibration significantly improved recall while maintaining strong precision.


## Key Insights

- Transfer learning with MobileNetV2 achieved strong performance despite modest dataset size
- Threshold optimization improved damage F1 from **0.80 → 0.93**
- Model successfully learned damage-relevant visual patterns
- Main failure modes occur with:
  - Very subtle damage
  - Reflective surfaces
  - Strong lighting artifacts

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Scikit-Learn**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**

---


## Future Improvements

- Upgrade to stronger backbones (EfficientNet / ConvNeXt)
- Introduce object detection / segmentation for localized damage detection
- Expand dataset with subtle-damage examples
- Deploy as Streamlit / FastAPI inspection tool

---

## Author

**Younes Soulaiman**  
AI / Data / Business Analytics Engineer

