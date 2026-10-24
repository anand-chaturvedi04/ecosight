
from pathlib import Path

DATASET = Path(r"D:\Data\ecosight_orig")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in SPLITS:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print(f"\n{split}: FOLDERS NOT FOUND")
        print("Expected:", image_dir)
        print("Expected:", label_dir)
        continue

    images = {
        p.stem for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = images - labels
    missing_images = labels - images

    bad_lines = 0
    invalid_classes = set()
    label_count = 0

    for label_file in label_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError("Expected 5 values")

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if not 0 <= class_id <= 10:
                    invalid_classes.add(class_id)

                if any(not 0 <= x <= 1 for x in coords):
                    raise ValueError("Coordinates outside 0–1")

                label_count += 1

            except ValueError as e:
                bad_lines += 1
                if bad_lines <= 10:
                    print(
                        f"Invalid annotation: {label_file.name} | "
                        f"{line!r} | Reason: {e}"
                    )

    print(f"\n--- {split.upper()} ---")
    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Annotated objects:", label_count)
    print("Images without matching labels:", len(missing_labels))
    print("Labels without matching images:", len(missing_images))
    print("Invalid annotation lines:", bad_lines)
    print("Unexpected class IDs:", sorted(invalid_classes))
