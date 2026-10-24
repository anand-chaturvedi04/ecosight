# 🌿 EcoSight AI

**AI-Powered Computer Vision-Based Smart Waste Segregation System for Sustainable Campus Waste Management**

EcoSight AI is an autonomous computer-vision waste segregation platform. It leverages **Ultralytics YOLOv8n** to classify waste items from camera snapshots or uploads, maps them into three distinct streams (**Organic/Compost**, **Recyclable**, and **General Waste**), flags uncertain predictions ($< 60\%$ confidence), simulates IoT smart bin actuators, and calculates real-time carbon offsets and Haversine routing to campus disposal stations.

---

## 🚀 Key Features

- **Multi-Class Detection:** Fine-tuned on the Roboflow Waste Detection v3 dataset across 11 classes, with automatic fallback to base COCO weights for live demonstration.
- **Three-Stream Mapping:** Strict segregation rules routing items to Organic, Recyclable, and General disposal streams.
- **Uncertainty Calibration:** Detections below $60\%$ confidence are labelled *Uncertain*, alerting users to inspect before disposal.
- **Dual Input Modes:** Supports both static photo uploads and **Live Camera Snapshots** via `st.camera_input`.
- **Virtual Smart Bin Actuator:** Visual flap animation and serial hardware commands (`SERVO:STREAM:OPEN:90`) ready for microcontroller/Arduino integration.
- **Eco-Impact & Carbon Tracker:** Real-time estimation of $CO_2$ emissions averted (g $CO_2e$) and natural decomposition duration.
- **Campus Geo-Bin Navigator:** Straight-line Haversine distance calculator routing users to the nearest compatible campus collection hubs.
- **Telemetry & Field Journal:** Historical scan logging to CSV with interactive distribution charts and confidence analytics.

---

## 🛠️ Quickstart Setup (Windows PowerShell)

```powershell
# 1. Navigate to the project directory
cd ecosight

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch (CUDA or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit application
streamlit run app.py
```

---

## 🧪 Training & Evaluation

```powershell
# Sanity check pilot run (3 epochs)
python train.py --mode pilot --data "<path_to_dataset_yaml>"

# Full training run with time budget
python train.py --mode full --hours 1.5

# Validation benchmark evaluation (generates results\metrics.json)
python evaluate.py
```

---

## 📊 Dataset & Attribution

- **Dataset:** [Waste Detection v3 (Roboflow Universe)](https://universe.roboflow.com/test-mq1p0/waste-detection-v3-tunsu/dataset/7), licensed under **CC BY 4.0**.
- **Bin Coordinates:** Illustrative demo locations around campus infrastructure for navigation demonstration.
