"""EcoSight AI - reproducible YOLO training script.

Usage (from the project folder, venv active):
  python train.py --mode pilot [--data "<dataset folder or data.yaml>"]
  python train.py --mode full --hours 1.5
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXPECTED_NAMES = [
    "Compost-Cup", "Compost-Lid", "Compost-Napkin", "Compost-Plate-Bowl", "Compost-Utencil",
    "Recycle-Aluminum-Can", "Recycle-Aluminum-Foil", "Recycle-Food-Packaging",
    "Recycle-Plastic-Bottle", "Trash-Bowl-Cup", "Trash-Wrapper",
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dataset_yaml(candidate: Path) -> Path | None:
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        return None
    hits = [h for h in sorted(candidate.rglob("data.yaml")) if ".venv" not in h.parts]
    return hits[0] if hits else None


def split_dir(root: Path, names: list[str]) -> Path | None:
    for n in names:
        p = root / n / "images"
        if p.is_dir():
            return p
    return None


def check_labels(images_dir: Path, nc: int) -> dict:
    labels_dir = images_dir.parent / "labels"
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    s = {"images": len(imgs), "missing_label_files": 0, "empty_label_files": 0, "boxes": 0,
         "polygon_lines": 0, "malformed_lines": 0, "bad_class_lines": 0, "out_of_range_lines": 0,
         "class_counts": [0] * nc}
    for img in imgs:
        lf = labels_dir / (img.stem + ".txt")
        if not lf.exists():
            s["missing_label_files"] += 1
            continue
        lines = [ln.split() for ln in lf.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            s["empty_label_files"] += 1
        for parts in lines:
            try:
                cid = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                s["malformed_lines"] += 1
                continue
            if len(parts) == 5:
                s["boxes"] += 1
            elif len(parts) > 5 and len(parts) % 2 == 1:
                s["polygon_lines"] += 1
            else:
                s["malformed_lines"] += 1
                continue
            if any(v < 0 or v > 1 for v in vals):
                s["out_of_range_lines"] += 1
            if 0 <= cid < nc:
                s["class_counts"][cid] += 1
            else:
                s["bad_class_lines"] += 1
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--data", default=str(ROOT / "data"), help="dataset folder or data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hours", type=float, default=None, help="time budget; overrides epochs")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    import ultralytics

    src_yaml = find_dataset_yaml(Path(args.data))
    if src_yaml is None:
        raise SystemExit(f"[ERROR] No data.yaml found under: {args.data}\n"
                         "Pass --data with the dataset folder (or copy the dataset into .\\data).")
    root = src_yaml.parent
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    print(f"[INFO] Dataset YAML: {src_yaml}\n[INFO] Classes ({len(names)}): {names}")
    if names != EXPECTED_NAMES:
        print("[WARN] Class names differ from the 11 expected names - update src/waste_logic.py mapping!")

    train_dir = split_dir(root, ["train"])
    val_dir = split_dir(root, ["valid", "val"])
    test_dir = split_dir(root, ["test"])
    if train_dir is None or val_dir is None:
        raise SystemExit(f"[ERROR] Could not find train/images and valid/images under {root}")

    stats = {"train": check_labels(train_dir, len(names)), "val": check_labels(val_dir, len(names))}
    if test_dir:
        stats["test"] = check_labels(test_dir, len(names))
    for k, v in stats.items():
        print(f"[CHECK] {k}: " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "class_counts"))
        print(f"        class_counts={v['class_counts']}")
    if stats["train"]["bad_class_lines"] or stats["train"]["malformed_lines"]:
        print("[WARN] Bad/malformed label lines found - Ultralytics will skip corrupt files.")

    out_yaml = ROOT / "data" / "ecosight.yaml"
    out_yaml.parent.mkdir(exist_ok=True)
    new_cfg = {"path": str(root.resolve()), "train": str(train_dir.relative_to(root)).replace("\\", "/"),
               "val": str(val_dir.relative_to(root)).replace("\\", "/"),
               "names": {i: n for i, n in enumerate(names)}}
    if test_dir:
        new_cfg["test"] = str(test_dir.relative_to(root)).replace("\\", "/")
    out_yaml.write_text(yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[INFO] Wrote {out_yaml}")

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[WARN] CUDA not available - training on CPU will be very slow. Run the pilot only.")
    print(f"[INFO] torch {torch.__version__}, ultralytics {ultralytics.__version__}, device={device}")

    results_dir = ROOT / "results"
    model = YOLO(args.model)
    t0 = time.time()
    common = dict(data=str(out_yaml), imgsz=args.imgsz, device=device, workers=2,
                  seed=args.seed, project=str(results_dir), plots=True)

    if args.mode == "pilot":
        model.train(epochs=2, batch=8, fraction=0.05, name="pilot", exist_ok=True, **common)
        print(f"\n[PILOT DONE] {time.time() - t0:.0f}s total. This only verifies the pipeline; "
              "pilot weights are NOT copied to models/ and must not be reported as trained.")
        return

    train_kwargs = dict(common, batch=args.batch, patience=15, name="train_full")
    if args.hours:
        train_kwargs["time"] = args.hours
    else:
        train_kwargs["epochs"] = args.epochs
    model.train(**train_kwargs)

    best = Path(model.trainer.best)
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / "best.pt")
    metrics = {k: float(v) for k, v in model.trainer.metrics.items()}
    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_minutes": round((time.time() - t0) / 60, 1),
        "base_model": args.model, "args": vars(args), "device": str(device),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "python": platform.python_version(), "best_weights_source": str(best),
        "dataset_checks": stats, "trainer_metrics_at_best": metrics,
    }
    (results_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Copied best weights to models\\best.pt\n[DONE] Summary: results\\train_summary.json")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()