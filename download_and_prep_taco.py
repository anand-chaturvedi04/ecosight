"""EcoSight AI - Automated TACO Dataset Downloader & YOLO Annotation Converter.

Downloads real-world litter images and converts COCO polygon/box annotations
into normalized YOLO detection format, partitioned into train/val splits.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Target Class Taxonomy (Maps 60 TACO categories into 7 balanced, high-utility classes)
TARGET_CLASSES = [
    "Bottle",           # 0 -> Recyclable
    "Can",              # 1 -> Recyclable
    "Cup",              # 2 -> Recyclable / Organic
    "Carton",           # 3 -> Recyclable
    "Wrapper-Bag",      # 4 -> General Waste
    "Cigarette",        # 5 -> General Waste
    "Other-Waste",      # 6 -> General Waste
]

CLASS_MAP: dict[str, int] = {
    # Bottles
    "Clear plastic bottle": 0, "Glass bottle": 0, "Other plastic bottle": 0,
    # Cans & Metals
    "Drink can": 1, "Food Can": 1, "Aerosol": 1, "Aluminium foil": 1,
    # Cups
    "Disposable plastic cup": 2, "Paper cup": 2, "Foam cup": 2, "Glass cup": 2, "Other plastic cup": 2,
    # Cartons
    "Corrugated carton": 3, "Drink carton": 3, "Egg carton": 3, "Meal carton": 3, "Other carton": 3, "Pizza box": 3,
    # Bags & Wrappers
    "Plastic film": 4, "Other plastic wrapper": 4, "Crisp packet": 4, "Garbage bag": 4,
    "Single-use carrier bag": 4, "Polypropylene bag": 4, "Produce bag": 4, "Cereal bag": 4,
    "Bread bag": 4, "Retort pouch": 4, "Six pack rings": 4,
    # Cigarettes
    "Cigarette": 5,
}
# Everything else maps to 6 ("Other-Waste")


def download_one(img_meta: dict, target_path: Path) -> bool:
    """Download single image with browser User-Agent and fallback URLs."""
    if target_path.exists() and target_path.stat().st_size > 1000:
        return True

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    urls = [img_meta.get("flickr_640_url"), img_meta.get("flickr_url")]
    urls = [u for u in urls if u]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read()
                if len(content) > 500:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = target_path.with_suffix(".tmp")
                    temp_path.write_bytes(content)
                    temp_path.replace(target_path)
                    return True
        except Exception:
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="Download and convert TACO dataset to YOLOv8 format.")
    parser.add_argument("--annotations", default=str(ROOT.parent / "TACO" / "data" / "annotations.json"))
    parser.add_argument("--max_images", type=int, default=300, help="Number of images to download (0 = all 1500)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split proportion (default: 0.2)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent download threads")
    parser.add_argument("--out_dir", default=str(ROOT / "data" / "taco_yolo"))
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        sys.exit(f"[ERROR] Annotations file not found: {ann_path}")

    print(f"[INFO] Loading annotations from {ann_path}...")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_anns: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        img_id_to_anns.setdefault(a["image_id"], []).append(a)

    images = coco["images"]
    if args.max_images > 0 and args.max_images < len(images):
        random.seed(42)
        images = random.sample(images, args.max_images)

    total = len(images)
    print(f"[INFO] Target download count: {total} images (using {args.workers} concurrent workers)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Download in parallel
    t0 = time.time()
    success_imgs = []

    print("[INFO] Downloading images from Cloudflare CDN...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for img in images:
            fname = Path(img["file_name"]).name
            target = out / "raw_images" / fname
            fut = executor.submit(download_one, img, target)
            futures[fut] = (img, target)

        done = 0
        for fut in as_completed(futures):
            img, target = futures[fut]
            done += 1
            if fut.result():
                success_imgs.append((img, target))
            if done % 25 == 0 or done == total:
                pct = int(done / total * 100)
                sys.stdout.write(f"\r[PROGRESS] Downloaded {len(success_imgs)}/{done} of {total} ({pct}%)")
                sys.stdout.flush()

    print(f"\n[INFO] Download finished in {time.time()-t0:.1f}s. Total verified: {len(success_imgs)} images.")

    # Partition into train / val
    random.seed(42)
    random.shuffle(success_imgs)
    n_val = max(1, int(len(success_imgs) * args.val_ratio))
    val_set = success_imgs[:n_val]
    train_set = success_imgs[n_val:]

    splits = {"train": train_set, "val": val_set}
    total_boxes = 0

    for split_name, img_list in splits.items():
        img_dir = out / "images" / split_name
        lbl_dir = out / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_meta, raw_file in img_list:
            # Copy or link image
            target_img = img_dir / raw_file.name
            target_lbl = lbl_dir / (raw_file.stem + ".txt")

            if not target_img.exists():
                import shutil
                shutil.copy2(raw_file, target_img)

            W = float(img_meta.get("width") or 640)
            H = float(img_meta.get("height") or 640)

            anns = img_id_to_anns.get(img_meta["id"], [])
            lines = []
            for a in anns:
                cat_name = cat_id_to_name.get(a["category_id"], "")
                target_cls = CLASS_MAP.get(cat_name, 6)  # 6 = Other-Waste

                bbox = a.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = bbox
                if w <= 0 or h <= 0 or W <= 0 or H <= 0:
                    continue

                xc = min(1.0, max(0.0, (x + w / 2.0) / W))
                yc = min(1.0, max(0.0, (y + h / 2.0) / H))
                nw = min(1.0, max(0.0, w / W))
                nh = min(1.0, max(0.0, h / H))

                lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                total_boxes += 1

            target_lbl.write_text("\n".join(lines), encoding="utf-8")

    # Generate YOLO data.yaml
    yaml_path = ROOT / "data" / "ecosight_taco.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    names_dict = {i: name for i, name in enumerate(TARGET_CLASSES)}
    yaml_content = f"""# EcoSight AI - TACO Multi-Class Waste Dataset
path: {out.resolve().as_posix()}
train: images/train
val: images/val
names:
""" + "\n".join(f"  {i}: {name}" for i, name in enumerate(TARGET_CLASSES)) + "\n"

    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[SUCCESS] Dataset prepared at: {out}")
    print(f"[SUCCESS] Train images: {len(train_set)} | Val images: {len(val_set)} | Bounding boxes: {total_boxes}")
    print(f"[SUCCESS] Dataset config generated: {yaml_path}")


if __name__ == "__main__":
    main()
