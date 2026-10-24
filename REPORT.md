# EcoSight AI - Project Report

## 1. Introduction
Campus waste is often mis-sorted. EcoSight AI detects waste items in photos, maps them to disposal categories and points users to the nearest compatible bin.

## 2. Objectives
Detect 11 waste classes; map them to Organic/Compost, Recyclable and General Waste; express uncertainty honestly; provide a bin finder, history and analytics in a Streamlit app.

## 3. Dataset
Waste Detection v3 (Roboflow Universe), CC BY 4.0: https://universe.roboflow.com/test-mq1p0/waste-detection-v3-tunsu/dataset/7
Classes: Compost-Cup, Compost-Lid, Compost-Napkin, Compost-Plate-Bowl, Compost-Utencil, Recycle-Aluminum-Can, Recycle-Aluminum-Foil, Recycle-Food-Packaging, Recycle-Plastic-Bottle, Trash-Bowl-Cup, Trash-Wrapper.
Split sizes and label checks: [FILL FROM results\train_summary.json -> dataset_checks]. An earlier check reported 2,631 training and 46 validation images and 102 flagged training annotation lines; the cause was not confirmed. No test split was found, so [STATE WHETHER A TEST SPLIT EXISTED IN THE CHECKS].

## 4. Methodology
**Model:** YOLOv8n (Ultralytics) initialised from COCO-pretrained weights and fine-tuned on the 11-class dataset (transfer learning).
**Training:** `train.py` validates labels, writes `data\ecosight.yaml`, runs a short pilot, then full training with image size 640, seed 42, early-stopping patience 15, default Ultralytics augmentation. Epochs/time budget and batch size: [FILL FROM results\train_full\args.yaml].
**Evaluation:** `evaluate.py` runs Ultralytics validation on the validation split and reports precision, recall, mAP@0.5 and mAP@0.5:0.95, overall and per class.
**Category mapping:** fixed rule table in `src/waste_logic.py` (see app About tab).
**Uncertainty:** detections below 0.25 are hidden; those below 0.60 are labelled "Uncertain" and the overall verdict asks for manual verification.
**Bin finder:** haversine straight-line distance to the nearest bin accepting the detected category, using 10 illustrative demo bins (not verified infrastructure).
**Analytics:** every analysis is appended to `results\history.csv`; charts are computed from these actual records.

## 5. Results
[FILL FROM results\metrics_summary.md. Copy the overall and per-class tables, the training plots from results\train_full\ (results.png, confusion_matrix.png, PR_curve.png) and the speed figures. Do not add numbers that are not in these files.]

## 6. Limitations
- Validation set is small (46 images in the earlier check), so metrics are noisy and also used for model selection; no independent test set unless one is added.
- Only 11 classes; unseen objects, clutter and lighting changes can cause errors.
- The category mapping is a fixed rule and may differ from campus policy.
- Bin locations and distances are illustrative.
- Not tested on real campus bins or live camera feeds.

## 7. Future Work
Held-out test split, more campus-specific images, live webcam mode, verified bin map, per-bin fill-level data.

## 8. Conclusion
[WRITE AFTER RESULTS: summarise actual measured performance and what was and was not completed.]

## 9. References
Ultralytics YOLOv8; Roboflow Waste Detection v3 dataset (CC BY 4.0).