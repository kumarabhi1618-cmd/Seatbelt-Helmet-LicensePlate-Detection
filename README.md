# 🚦 TrafficGuard AI

**Automated Traffic Violation Detection System**

> An end-to-end computer vision pipeline that detects helmet and seatbelt violations from traffic images, identifies license plates, and generates downloadable violation reports — all in a real-time Streamlit web app.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://traffic-guard-ai-cmuwavclgnb4egireb6cl5.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-orange)
![EE655](https://img.shields.io/badge/Course-EE655-lightgrey)

---

## 📌 Overview

TrafficGuard AI is a four-model deep learning pipeline designed for automated traffic law enforcement. Given a single traffic image, the system:

1. Detects all vehicles (cars and motorcycles) in the frame
2. Checks motorcycles for helmet compliance
3. Checks cars for seatbelt compliance
4. Detects and crops the license plate of any violating vehicle
5. Reads the plate text using OCR
6. Displays structured violation records with confidence scores and a downloadable report

---

## 🔍 Detection Pipeline

```
Image ──► Model 1: Vehicle Detection
            ├─ motorcycle ──► Model 3: Helmet Detection
            │                   WithoutHelmet ──► Model 4: License Plate ──► EasyOCR ──► Violation Record
            │                   WithHelmet    ──► ✅ Compliant
            └─ car ──────► Model 2: Seatbelt Detection
                              NoSeatbelt ──► Model 4: License Plate ──► EasyOCR ──► Violation Record
                              Seatbelt   ──► ✅ Compliant
```

---

## 🤖 Models

| # | File | Task | Base Model | Classes | Epochs |
|---|------|------|------------|---------|--------|
| 1 | `model1_vehicle_detection.pt` | Multiclass vehicle detection | YOLOv8m | `car`, `motorcycle` (+ 15 others) | 80 |
| 2 | `model2_seatbelt.pt` | Seatbelt detection | YOLOv8n | `Seatbelt`, `NoSeatbelt` | 60 |
| 3 | `model3_helmet.pt` | Helmet detection + plate assist | YOLOv8m | `WithHelmet`, `WithoutHelmet`, `Motorcycle`, `LisencePlate` | 80 |
| 4 | `model4_license_plate.pt` | License plate localization | YOLOv8n | `LisencePlate` | 80 |

All models are trained on Kaggle using GPU, exported as `.pt` weight files, and loaded at inference time via [Ultralytics](https://github.com/ultralytics/ultralytics).

---

## 🗂️ Training Notebooks

Each model has its own Kaggle training notebook:

| Notebook | Model Trained |
|----------|--------------|
| `Multiclass-Vehicle-Detection-Trained.ipynb` | Model 1 — Vehicle Detection |
| `Seatbelt-Detection-Trained.ipynb` | Model 2 — Seatbelt Detection |
| `Helmet-Detection-Trained.ipynb` | Model 3 — Helmet Detection |
| `License-Plate-Detection-Trained.ipynb` | Model 4 — License Plate Detection |

### Training Configuration

**Model 1 — Vehicle Detection**
- Dataset: `roadvision_blur` (HeteroTraffic Annotated Dataset, multi-class)
- Split: 80 / 12 / 8 (train / val / test)
- Augmentation: HSV jitter, horizontal flip, mosaic

**Model 2 — Seatbelt Detection**
- Dataset: `Cleaned_Dataset_Seatbelt_Detection`
- Augmentation: Low mosaic (0.5) since seatbelt is a small region; scale, translate, copy-paste

**Model 3 — Helmet Detection**
- Dataset: `helmet-detection-3`
- Augmentation: Increased scale sensitivity for small head regions; mosaic 0.7, copy-paste

**Model 4 — License Plate Detection**
- Dataset: `Number_Plate_Detection`
- Augmentation: Conservative (plates rarely tilted); partial occlusion via erasing; low mosaic

---

## 🖥️ App Features (`app.py`)

- **Image upload** — supports JPG, JPEG, PNG, BMP, WEBP
- **Adjustable confidence thresholds** — individual sliders for each of the four models (sidebar)
- **Motorcycle crop extension** — configurable upward extension (%) to capture rider + helmet above the detected vehicle box
- **Annotated output image** — bounding boxes colour-coded by result (green = compliant, red = violation, orange = uncertain)
- **Violation records** — per-violation cards showing vehicle type, offence, license plate text, and confidence scores for detection + OCR
- **Compliant vehicle summary** — lists all vehicles with no detected violation
- **Downloadable report** — exports a plain-text `.txt` violation report

### Summary Metrics Displayed

| Metric | Description |
|--------|-------------|
| Vehicles Detected | Total cars + motorcycles found |
| Violations Found | Count of helmet/seatbelt violations |
| Compliant | Vehicles with no detected violation |
| No Helmet | Motorcycle violations |
| No Seatbelt | Car violations |
| Inference Time | End-to-end pipeline duration (seconds) |

---

## 🔧 OCR Pipeline

License plate crops are enhanced before OCR:
1. Grayscale conversion
2. Upscaling (min 180px height)
3. CLAHE contrast enhancement
4. Otsu thresholding
5. EasyOCR with alphanumeric allowlist (`A-Z 0-9 -`)
6. Multi-result fusion (sorted left-to-right, mean confidence)

For motorcycles, plates are sourced from both Model 3 (helmet model also detects plates) and Model 4 — the highest-confidence detection is used.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit ultralytics easyocr opencv-python-headless Pillow numpy
```

### Run Locally

1. Clone the repository and place the four model `.pt` files in the root directory:
   ```
   model1_vehicle_detection.pt
   model2_seatbelt.pt
   model3_helmet.pt
   model4_license_plate.pt
   ```

2. Launch the app:
   ```bash
   streamlit run app.py
   ```

3. Open `http://localhost:8501` in your browser.

### Streamlit Cloud Deployment

The app is deployment-ready for Streamlit Cloud. Ensure `requirements.txt` includes:

```
streamlit
ultralytics
easyocr
opencv-python-headless
Pillow
numpy
```

---

## 📁 Project Structure

```
TrafficGuard-AI/
├── app.py                                      # Main Streamlit application
├── model1_vehicle_detection.pt                 # YOLOv8 vehicle detection weights
├── model2_seatbelt.pt                          # YOLOv8 seatbelt detection weights
├── model3_helmet.pt                            # YOLOv8 helmet detection weights
├── model4_license_plate.pt                     # YOLOv8 license plate detection weights
├── Multiclass-Vehicle-Detection-Trained.ipynb  # Training notebook — Model 1
├── Seatbelt-Detection-Trained.ipynb            # Training notebook — Model 2
├── Helmet-Detection-Trained.ipynb              # Training notebook — Model 3
├── License-Plate-Detection-Trained.ipynb       # Training notebook — Model 4
└── requirements.txt                            # Python dependencies
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | [Streamlit](https://streamlit.io/) |
| Object Detection | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Image Processing | OpenCV, Pillow |
| Training Platform | Kaggle (GPU) |
| Deployment | Streamlit Cloud |

---

## ⚙️ Configuration (Sidebar)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Vehicle confidence | 0.40 | 0.20–0.90 | Minimum score for vehicle detections |
| Helmet confidence | 0.40 | 0.20–0.90 | Minimum score for helmet/no-helmet |
| Seatbelt confidence | 0.40 | 0.20–0.90 | Minimum score for seatbelt/no-seatbelt |
| Plate confidence | 0.35 | 0.15–0.90 | Minimum score for plate localization |
| Moto crop extension | 60% | 20–120% | % of box height added above motorcycle to capture rider |

---

## 📄 License

This project was developed as part of **EE655** coursework. All datasets used for training are sourced from Kaggle and retain their respective licenses.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection backbone
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for plate text reading
- Kaggle datasets: HeteroTraffic Annotated Dataset, Seatbelt Detection Dataset, Helmet Detection Dataset, License Plate Detection Dataset
