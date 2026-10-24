from pathlib import Path
import shutil

src = Path(r"D:\Data\ecosight_orig")
dst = Path(r"D:\Data\ecosight_detect_fixed")

if dst.exists():
    raise SystemExit(
        f"STOP: {dst} already exists. Nothing was changed."
    )

print("Copying dataset...")
shutil.copytree(src, dst)

converted = 0

for label_file in (dst / "train" / "labels").glob("*.txt"):
    lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    output = []

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        if len(parts) == 5:
            output.append(line)
            continue

        if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
            try:
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if not all(0 <= x <= 1 for x in coords):
                    raise ValueError("Coordinates outside 0..1")

                xs = coords[0::2]
                ys = coords[1::2]

                if len(xs) < 3 or len(xs) != len(ys):
                    raise ValueError("Invalid polygon")

                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)

                if xmax <= xmin or ymax <= ymin:
                    raise ValueError("Zero-area box")

                xc = (xmin + xmax) / 2
                yc = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin

                output.append(
                    f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                )
                converted += 1
            except (ValueError, IndexError) as exc:
                raise SystemExit(
                    f"Invalid polygon in {label_file.name}: {line}\n{exc}"
                )
        else:
            raise SystemExit(
                f"Unexpected label format in {label_file.name}: {line}"
            )

    label_file.write_text(
        "\n".join(output) + ("\n" if output else ""),
        encoding="utf-8"
    )

print(f"Done. Converted polygon lines: {converted}")
print(f"Original untouched: {src}")
print(f"Detection copy: {dst}")
