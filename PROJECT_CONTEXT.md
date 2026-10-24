# EcoSight AI — Project Context

## 1. Project Title
AI-Powered Computer Vision-Based Smart Waste Segregation System for Sustainable Campus Waste Management.

## 2. Goal
Build a working, polished, and technically credible AI waste-segregation prototype for a BTech CSE college project. Prioritize a reliable end-to-end demonstration, useful features, and an accurate project report.

## 3. Environment
- OS: Windows
- Project folder: `C:\Users\Asus\Desktop\EcoSight`
- Python: 3.14.0
- RAM: 16 GB
- GPU: NVIDIA RTX 2050 (4 GB VRAM) and Intel Iris Xe
- Existing folders: `.venv`, `data`, `models`, `results`, `src`
- Existing files include `app.py`, `check_dataset.py`, `README.md`, `REPORT.md`, and `requirements.txt`.
- Inspect existing files and installed packages before changing anything. Reuse working code and avoid duplicate implementations.

## 4. Dataset
Source: https://universe.roboflow.com/test-mq1p0/waste-detection-v3-tunsu/dataset/7

The dataset's YAML specifies the CC BY 4.0 license. Provide appropriate attribution.

The dataset was previously located at:
`C:\Users\Asus\OneDrive\画像\Desktop\Projects`

Its known classes are:
1. Compost-Cup
2. Compost-Lid
3. Compost-Napkin
4. Compost-Plate-Bowl
5. Compost-Utencil
6. Recycle-Aluminum-Can
7. Recycle-Aluminum-Foil
8. Recycle-Food-Packaging
9. Recycle-Plastic-Bottle
10. Trash-Bowl-Cup
11. Trash-Wrapper

A previous dataset check reported 2,631 training images and 46 validation images. It also flagged 102 training annotation lines, but their cause has not been confirmed. A test folder was not found in the previously checked dataset location.

These are dataset counts, not model performance results. Verify actual paths and files before relying on them.

## 5. Required Features
1. Image upload and object detection with bounding boxes, class names, and confidence scores.
2. Map the 11 dataset classes into Organic/Compost, Recyclable, and General Waste categories.
3. Handle low-confidence predictions honestly and show uncertainty where appropriate.
4. A campus bin finder with 8–10 illustrative bin locations and nearest-compatible-bin calculation. Clearly label demo coordinates as illustrative, not verified campus infrastructure.
5. Analysis history and simple analytics based on actual detections.
6. A clean, polished, easy-to-demonstrate Streamlit interface.
7. Clear setup instructions and run commands.
8. A project report documenting the actual implementation, methodology, results, limitations, and future improvements.

## 6. Engineering Requirements
- Inspect the existing project before coding.
- Check whether trained model weights such as `best.pt` actually exist.
- Do not falsely claim a generic pretrained detector is trained specifically for waste.
- If no suitable trained model exists, choose the most practical reliable approach for the available time and clearly explain its limitations.
- Do not invent training results, accuracy, precision, recall, or other metrics.
- Avoid unnecessary dependencies and long training runs.
- Check compatibility with the actual Python environment.
- Keep code organized, readable, and beginner-friendly.
- Fix critical execution errors before adding optional features.
- Do not overwrite useful existing work without inspecting it first.

## 7. Deadline and Priorities
The prototype, report, and submission must be completed urgently.

Prioritize:
1. A working application.
2. Reliable detection and clear limitations.
3. A polished demonstration.
4. An accurate report and setup instructions.
5. Additional enhancements only if time permits.

Work efficiently and minimize unnecessary questions. Inspect the available files first, make the maximum safe progress, and report exactly what was implemented, tested, and left incomplete.