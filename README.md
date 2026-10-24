# EcoSight AI

Computer-vision waste segregation for campus waste management: YOLOv8n detection of 11 waste classes, mapped to Organic/Compost, Recyclable and General Waste, with a Streamlit UI, nearest-bin finder (illustrative demo locations) and analytics.

## Setup (Windows PowerShell)
```powershell
cd C:\Users\Asus\Desktop\EcoSight
.\.venv\Scripts\Activate.ps1
python --version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -r requirements.txt
```
If no PyTorch wheel exists for your Python version (3.14), create a Python 3.12 venv instead: `py -3.12 -m venv .venv312`, activate it and repeat the steps above.

## Train and evaluate
```powershell
python train.py --mode pilot --data "<dataset folder>"   # pipeline check only
python train.py --mode full --hours 1.5                   # real training, saves models\best.pt
python evaluate.py                                        # writes results\metrics.json
```
Omit `--data` if the dataset is already under `data\`.

## Run the app
```powershell
streamlit run app.py
```

## Outputs
`models\best.pt`, `results\train_full\` (results.csv, plots, args.yaml), `results\train_summary.json`, `results\metrics.json`, `results\metrics_summary.md`, `results\history.csv`.

## Attribution
Dataset: Waste Detection v3 (Roboflow Universe), CC BY 4.0 - https://universe.roboflow.com/test-mq1p0/waste-detection-v3-tunsu/dataset/7
Bin coordinates are illustrative demo data, not verified campus infrastructure.