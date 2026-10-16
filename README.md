# Micro-level-Surface-Crack-Detection-In-Steel
Deep-learning pipeline that detects and classifies surface defects on hot-rolled steel strips, with Grad-CAM explainability and a real-time inspection portal.

> AI course project — Thapar Institute of Engineering and Technology, Patiala
> Submitted to **Dr. Ritesh Sharma**

## Team

| Name | Roll No. |
|---|---|
| Samriddhi Gupta (Group Leader) | 1024170165 |
| Waris | 1024170155 |
| Tanishq | 1024170156 |
| Kanan | 1024170164 |
| Ananya Dayal | 1024170205 |

## Why

Surface quality affects structural integrity in automotive, aerospace and construction parts. Micro-cracks under stress can lead to component failure. Manual inspection is slow and inconsistent. A trained model runs 24/7, gives the same answer every time, and improves as it's retrained.

## Results

| Model | Test Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---|---|---|
| SimpleCNN (from scratch) | 97.0% | 97.0% | 97.0% | 99.9% (macro 0.9986) |
| **EfficientNet-B0 (transfer learning)** | **98.5%** | **98.5%** | **98.5%** | **100.0%** |

- **SimpleCNN:** weakest class is pitted surface at 0.91 recall. Patches and rolled-in scale reach 1.00.
- **EfficientNet-B0:** every class is at 1.00 recall except inclusion (0.91), which is sometimes confused with scratches.
- **Synthetic crack dataset (CrackCNN):** 6 classes including *crack* and *normal*. Macro AUC is 0.953. Crack is detected at 0.98 recall. Inclusion is the hardest class (0.56).

## Dataset — NEU Surface Defect Database

- **Images:** 1,800 grayscale images at 200 × 200 px
- **Classes:** 6 balanced classes with 300 images each: crazing, inclusion, patches, pitted surface, rolled-in scale and scratches
- **Split:** stratified 70% train / 15% validation / 15% test

## Pipeline

```
Dataset → Cleaning → Preprocessing & Augmentation → CNN Features
        → Training & Optimisation → Validation Metrics → Prediction → Grad-CAM Verification
```

**Cleaning:** these steps were designed for real factory camera images.

- Gaussian blur to remove noise
- Morphological opening/closing to remove artifacts
- Histogram equalisation to even out contrast
- Edge-based ROI extraction to isolate the steel strip

**Preprocessing:**

- Resize to 128 × 128 (224 × 224 for EfficientNet)
- RGB normalisation with mean = std = [0.5, 0.5, 0.5]

**Augmentation:** each transform below is applied randomly, with probability p where given.

- Horizontal flip (p=0.5) and vertical flip (p=0.3)
- 90° rotation (p=0.5) and ±15° rotation
- CLAHE
- Gaussian noise
- Grid and elastic distortion
- Brightness/contrast changes
- CoarseDropout
- Gaussian blur

## Models

### SimpleCNN

- **Backbone:** 4 convolutional blocks with 32 → 64 → 128 → 256 filters. Each block is Conv2d → BatchNorm → ReLU, repeated twice, then MaxPool. It produces 8×8×256 feature maps.
- **Head:** Global Average Pooling → Dropout 0.4 → FC 256→512 (ReLU) → Dropout 0.2 → FC 512→6 → Softmax

### EfficientNet-B0

- **Backbone:** ImageNet-pretrained, frozen in phase 1. It uses compound scaling, MBConv blocks, squeeze-and-excitation and SiLU activations.
- **Size:** about 5.3M parameters
- **Head:** GAP → Dropout 0.3 → Dense 256 (ReLU) → Dropout 0.3 → Dense 6 (Softmax)

### Training configuration

| Setting | Value |
|---|---|
| Epochs | 30 |
| Batch size | 32 |
| Optimizer | AdamW (lr 1e-3, β = 0.9 / 0.999, weight decay 1e-4) |
| LR scheduler | CosineAnnealingLR (1e-3 → 1e-6) |
| Loss | Categorical cross-entropy |
| Regularisation | Dropout + BatchNorm |
| Early stopping | Patience 7 on validation loss |
| Checkpointing | Keeps the best model by validation loss |
| Device | CUDA |

## Explainability — Grad-CAM

Grad-CAM heatmaps show which regions drove each prediction:

- **Crazing:** linear fracture lines
- **Inclusion:** circular zones around embedded material
- **Patches:** broad irregular texture regions
- **Pitted surface:** clustered pit locations

The heatmaps stay consistent across augmented variants (flips, CLAHE, noise, dropout). This suggests the model focuses on the defects themselves, not on artifacts.

## Deployment

**NEU-STEEL Live Inspection Portal (built):**

- Upload a steel image or capture one from a camera
- See a Grad-CAM overlay
- See per-class confidence, the top-2 predictions and a pass/fail diagnosis

**Proposed production layer (not yet built):**

- **Edge AI:** NVIDIA Jetson with a target of under 50 ms latency and over 20 FPS on-premise
- **Smart factory integration:** PLC/SCADA integration with automatic pass/reject signals to conveyors
- **Web/API:** a Flask/FastAPI dashboard and REST endpoints for ERP systems
- **Analytics:** defect trends, batch quality reports and PDF exports for audits

## Getting Started

```bash
git clone https://github.com/warismittal/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Download the [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm) and place it under `data/`. Then run the training notebook or script.

## Tech Stack

Python · PyTorch / TensorFlow-Keras · Albumentations · OpenCV · scikit-learn · Matplotlib · Grad-CAM
