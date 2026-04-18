# 🚦 TrafficGuard AI

Automated traffic violation detection system built for the EE655 project.

# Link for the app

https://traffic-guard-ai-cmuwavclgnb4egireb6cl5.streamlit.app/

## Pipeline

```
Image ──► model1 (vehicle detection)
            ├─ motorcycle ──► model3 (helmet model)
            │                 └─ WithoutHelmet ──► model4 (plate) ──► OCR ──► Violation
            └─ car ──────► model2 (seatbelt)
                            └─ NoSeatbelt ──► model4 (plate) ──► OCR ──► Violation
```

## Model Classes

| Model | File | Classes |
|-------|------|---------|
| Vehicle Detection | `model1_vehicle_detection.pt` | `car`, `motorcycle` |
| Seatbelt | `model2_seatbelt.pt` | `Seatbelt`, `NoSeatbelt` |
| Helmet | `model3_helmet.pt` | `LisencePlate`, `Motorcycle`, `WithHelmet`, `WithoutHelmet` |
| License Plate | `model4_license_plate.pt` | `LisencePlate` |

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. Push all files (including `.pt` model files) to your GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file as `app.py`
4. Deploy

> ⚠️ Model `.pt` files must be in the **root** of the repo alongside `app.py`.

## Output

- Annotated image with bounding boxes and confidence scores
- Per-violation record with:
  - Violation type & confidence
  - Vehicle crop (evidence photo)
  - License plate crop
  - OCR-read plate number with confidence
- Downloadable `.txt` report
