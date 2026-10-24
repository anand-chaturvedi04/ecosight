from pathlib import Path
import yaml
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    data_dir = Path(r"D:\Data\ecosight_detect_fixed")
    output_dir = Path(r"D:\Data\ecosight_training")

    # 1. Verify the dataset configuration.
    source_yaml = data_dir / "data.yaml"

    if not source_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {source_yaml}"
        )

    config = yaml.safe_load(
        source_yaml.read_text(encoding="utf-8")
    )

    # 2. Use absolute dataset location and the available splits.
    config["path"] = str(data_dir)
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config.pop("test", None)

    for split in ("train", "valid"):
        image_dir = data_dir / split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image folder not found: {image_dir}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    training_yaml = output_dir / "ecosight.yaml"
    training_yaml.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    # 3. Load pretrained YOLOv8 Nano weights.
    pretrained = root / "yolov8n.pt"
    if not pretrained.exists():
        raise FileNotFoundError(
            f"Pretrained weights not found: {pretrained}"
        )

    model = YOLO(str(pretrained))

    # 4. Train into a NEW run directory.
    model.train(
        data=str(training_yaml),
        epochs=25,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(output_dir / "runs"),
        name="bottle_wrapper_v2",
        exist_ok=False,
        patience=7,
        cache=False,
        plots=True,
    )

    print("\nTraining finished.")
    print(
        "Check: D:\\Data\\ecosight_training\\runs"
        "\\bottle_wrapper_v2\\weights\\best.pt"
    )


if __name__ == "__main__":
    main()