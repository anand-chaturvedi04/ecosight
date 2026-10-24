"""Evaluate models/best.pt on the validation (or test) split and save real metrics."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "ecosight.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"[ERROR] Weights not found: {args.weights}. Run train.py --mode full first.")
    if not Path(args.data).exists():
        raise SystemExit(f"[ERROR] {args.data} not found. Run train.py first (it creates it).")

    import ultralytics
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    split_path = Path(cfg["path"]) / cfg[args.split]
    n_images = len([p for p in split_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, project=str(ROOT / "results"),
                  name=f"eval_{args.split}", exist_ok=True, plots=True, verbose=False)

    per_class = []
    present = set()
    for i, cls_idx in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        present.add(int(cls_idx))
        per_class.append({"class": m.names[int(cls_idx)], "precision": float(p), "recall": float(r),
                          "mAP50": float(ap50), "mAP50-95": float(ap)})
    missing = [n for i, n in m.names.items() if i not in present]

    out = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": args.weights, "split": args.split, "num_images": n_images,
        "ultralytics": ultralytics.__version__,
        "overall": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                    "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)},
        "per_class": per_class, "classes_without_instances_in_split": missing,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in m.speed.items()},
    }
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    o = out["overall"]
    md = [f"# Evaluation summary ({args.split} split, {n_images} images)", "",
          f"Weights: `{args.weights}` | Evaluated: {out['evaluated_at']}", "",
          "| Metric | Value |", "|---|---|",
          f"| Precision | {o['precision']:.3f} |", f"| Recall | {o['recall']:.3f} |",
          f"| mAP@0.5 | {o['mAP50']:.3f} |", f"| mAP@0.5:0.95 | {o['mAP50-95']:.3f} |", "",
          "| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
    md += [f"| {c['class']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['mAP50']:.3f} | {c['mAP50-95']:.3f} |"
           for c in per_class]
    if missing:
        md += ["", "Classes with no instances in this split (not evaluated): " + ", ".join(missing)]
    md += ["", f"Speed (ms/image): {out['speed_ms_per_image']}"]
    (res / "metrics_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("\n[DONE] Saved results\\metrics.json and results\\metrics_summary.md")


if __name__ == "__main__":
    main()