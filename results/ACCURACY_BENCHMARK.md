# EcoSight AI — Large-Scale Dataset Testing Report

**Evaluation Date:** October 24, 2026  
**Hardware Profile:** Intel Core i7-13620H CPU (Inference via ONNX Runtime C++ backend)  
**Dataset:** TACO (Trash Annotations in Context) Expanded Test Split  
- **Test Images:** 116 images  
- **Ground-Truth Labeled Items:** 324 objects  
  - Bottles: 37 | Cans: 27 | Cups: 18 | Cartons: 22 | Wrappers/Bags: 57 | Cigarettes: 37 | Clutter: 126  

---

## 1. Multi-Model Benchmark Results

| Model Architecture | Parameter Count | Inference Latency | Waste Precision | Waste Recall | F1 Score |
|---|---|---|---|---|---|
| **YOLOv8n Generic COCO Base** | 3.2M | 260.8 ms/frame | 28.6% | 5.4% | 0.091 |
| **Fine-Tuned Waste Model (HuggingFace)** | 11.1M | 139.1 ms/frame | 27.3% | 8.1% | 0.125 |

---

## 2. Key Insights for Faculty Review

1. **Why Generic Models Struggle on Wild Litter:**  
   Wild litter in public spaces is heavily deformed, occluded by leaves/dirt, or crushed flat. Standard COCO models trained on upright household objects miss over 90% of severely crushed cans and wrappers.
2. **Phase 2 Training Advantage:**  
   We now have 750 local TACO training images and 3,272 bounding boxes ready on disk (data/taco_yolo/). Training specifically on these deformed objects in Phase 2 will bridge the domain gap.
