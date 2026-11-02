# ScanScope: Multi-Model Medical Imaging Research Prototype

> ⚠️ **EDUCATIONAL RESEARCH DEMO ONLY**  
> This application cannot diagnose, screen for, or rule out cancer or any other disease.  
> Do not use it to make health decisions. Results shown are non-clinical model outputs for research and learning purposes only.

---

## Overview

**ScanScope** demonstrates how an educational web application routes different supported medical image types to distinct, specialized machine-learning models.

- **Strict Task Routing:** The router dispatches only to models explicitly registered and verified for the chosen examination type.
- **Explicit In-Scope Definition:** Selecting an examination type specifies both the modality and the precise prediction task. Unsupported modalities are never sent to mismatched models.
- **Safety First:** Output language strictly adheres to research labels, highlights known limitations, labels model scores clearly as non-calibrated, and does not provide clinical reassurance or triage advice.
- **Privacy Respecting:** User uploads are validated, processed locally in volatile storage, and immediately removed.

---

## Project Structure

```
anti gravity/
├── app.py                         # Flask web application & API routing
├── config.py                      # Application configuration & paths
├── requirements.txt               # Python package dependencies
├── models/
│   ├── base.py                    # Abstract BaseModel, ModelMetadata, PredictionResult
│   ├── registry.py                # Model registry and routing dispatcher
│   ├── histopathology.py          # Real Camelyon16 ResNet-18 patch classifier
│   └── chest_xray.py              # Chest X-ray classifier shell & loader
├── preprocessing/
│   └── transforms.py              # Normalized TorchVision transforms per modality
├── validation/
│   └── input_validator.py         # File type, dimension, size, and header validation
├── static/
│   ├── css/style.css              # Accessible, responsive styling (light/dark mode)
│   ├── js/app.js                  # Frontend state machine and interactive UI
│   └── samples/                   # Demo image samples for non-clinical testing
├── templates/
│   └── index.html                 # UI template with research notice banner
└── training/
    ├── train_chest_xray.py        # Complete training & held-out evaluation script
    └── generate_sample_patches.py # Synthetic sample generator for UI testing
```

---

## Implemented Models & Status

### 1. Histopathology — Tumor Metastasis Detection (`histopathology_tumor_detection`)
- **Status:** **Implemented & Fully Ready**
- **Architecture:** ResNet-18 (Baidu Research NCRF baseline)
- **Dataset:** Camelyon16 Challenge (Sentinel lymph node whole-slide images)
- **Labels:** `Tumor`, `Normal`
- **Weights:** Sibling repository checkpoint (`NCRF/ckpt/resnet18_base.ckpt`) loaded via PyTorch CPU.
- **Metrics Note:** Original study reported FROC / patch accuracy; no fabricated local test metrics are simulated.

### 2. Chest X-Ray — Pneumonia Detection (`chest_xray_pneumonia`)
- **Status:** **Shell Registered / Not Yet Trained** (Honest status displayed in UI)
- **Dataset:** Kermany et al. pediatric chest radiographs (CC BY 4.0)
- **Labels:** `NORMAL`, `PNEUMONIA`
- **Training Script:** Ready at `training/train_chest_xray.py` to train when the dataset is downloaded.

---

## Quickstart

### 1. Activate the Virtual Environment
```powershell
cd "c:\Users\Asus NL\Desktop\coding\ml\college mini project\anti gravity"
.\venv\Scripts\Activate.ps1
```

### 2. Run the Application
```powershell
python app.py
```
Open your browser to: [http://127.0.0.1:5000](http://127.0.0.1:5000)
