from pathlib import Path
from collections import Counter
import math
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit(
        "Pillow is required. Install it in your project environment with:\n"
        ".venv/bin/python -m pip install Pillow"
    )

ROOT = Path("/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26")
DATASET = ROOT / "final_cleaned_dataset_no_conflicts"
OUTPUT_YAML = ROOT / "data_final_cleaned.yaml"
REPORT = ROOT / "preprocessing_readiness_report.txt"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_CLASS_NAMES = ["pothole"]

def number_is_valid(s):
    try:
        return math.isfinite(float(s))
    except (TypeError, ValueError):
        return False

def main():
    if not DATASET.is_dir():
        sys.exit(f"ERROR: Cleaned dataset not found:\n{DATASET}")

    results = {}
    errors = []
    warnings = []
    all_sizes = Counter()
    total_boxes = 0
    total_empty = 0

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing images/labels directory")
            continue

        images = sorted(p for p in image_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        labels = list(label_dir.glob("*.txt"))
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing = sorted(image_stems - label_stems)
        orphan = sorted(label_stems - image_stems)
        sizes = Counter()
        box_count = 0
        empty_count = 0
        class_ids = Counter()
        annotation_errors = []

        for image_path in images:
            try:
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    size = im.size
                    im.load()
                sizes[size] += 1
                all_sizes[size] += 1
            except Exception as exc:
                errors.append(f"Unreadable image {image_path}: {exc}")
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            try:
                content = label_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                errors.append(f"Unreadable label {label_path}: {exc}")
                continue

            if not content:
                empty_count += 1
                continue

            for line_no, line in enumerate(content.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5 or not parts[0].isdigit() or not all(number_is_valid(x) for x in parts[1:]):
                    annotation_errors.append(f"{label_path.name}:{line_no}: malformed YOLO row")
                    continue
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                if cls != 0:
                    annotation_errors.append(f"{label_path.name}:{line_no}: class ID {cls}; expected 0")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    annotation_errors.append(f"{label_path.name}:{line_no}: values outside YOLO range")
                if x - w/2 < -1e-6 or y - h/2 < -1e-6 or x + w/2 > 1 + 1e-6 or y + h/2 > 1 + 1e-6:
                    annotation_errors.append(f"{label_path.name}:{line_no}: box extends beyond image")
                class_ids[cls] += 1
                box_count += 1

        if missing:
            errors.append(f"{split}: {len(missing)} images lack labels")
        if orphan:
            errors.append(f"{split}: {len(orphan)} orphan label files")
        if annotation_errors:
            errors.extend(f"{split}: {e}" for e in annotation_errors[:50])
            if len(annotation_errors) > 50:
                warnings.append(f"{split}: {len(annotation_errors)-50} more annotation errors omitted from report")

        results[split] = {
            "images": len(images),
            "labels": len(labels),
            "backgrounds": empty_count,
            "annotated": len(images) - len(missing) - empty_count,
            "boxes": box_count,
            "sizes": sizes,
            "class_ids": class_ids,
        }
        total_boxes += box_count
        total_empty += empty_count

    if set(results) != set(SPLITS):
        errors.append("Not all three splits could be inspected.")

    # Configuration is written only when structural checks pass.
    if not errors:
        yaml_text = (
            f"path: '{DATASET}'\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            "nc: 1\n"
            "names: ['pothole']\n"
        )
        OUTPUT_YAML.write_text(yaml_text, encoding="utf-8")
    else:
        warnings.append("Final YAML was NOT written because validation errors were found.")

    lines = [
        "PREPROCESSING READINESS REPORT",
        "=" * 60,
        f"Dataset: {DATASET}",
        "No images were resized, recompressed, moved, or deleted by this script.",
        "",
    ]
    for split in SPLITS:
        if split not in results:
            continue
        r = results[split]
        lines.extend([
            split.upper(),
            "-" * 40,
            f"Images: {r['images']}",
            f"Label files: {r['labels']}",
            f"Annotated images: {r['annotated']}",
            f"Empty-label/background images: {r['backgrounds']}",
            f"Bounding boxes: {r['boxes']}",
            f"Class IDs: {dict(r['class_ids'])}",
            "Most common image dimensions:",
        ])
        lines.extend([f"  {w}x{h}: {count}" for (w, h), count in r["sizes"].most_common(10)])
        lines.append("")

    lines.extend([
        "OVERALL",
        "=" * 60,
        f"Total images: {sum(r['images'] for r in results.values())}",
        f"Total boxes: {total_boxes}",
        f"Total background images: {total_empty}",
        f"Distinct image dimensions: {len(all_sizes)}",
        f"Final YAML written: {'YES' if not errors else 'NO'}",
        f"YAML path: {OUTPUT_YAML}",
        "",
        "ERRORS:",
        *(errors if errors else ["None."]),
        "",
        "WARNINGS:",
        *(warnings if warnings else ["None."]),
        "",
        "Note: this checks dataset structure and annotation format, not the visual correctness of every pothole box.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("PREPROCESSING READINESS CHECK FINISHED")
    for split in SPLITS:
        if split in results:
            r = results[split]
            print(f"{split.upper()}: images={r['images']}, labels={r['labels']}, backgrounds={r['backgrounds']}, boxes={r['boxes']}")
    print(f"Total images: {sum(r['images'] for r in results.values())}")
    print(f"Total boxes: {total_boxes}")
    print(f"Distinct image dimensions: {len(all_sizes)}")
    print(f"Errors: {len(errors)}")
    print(f"Report: {REPORT}")
    if not errors:
        print(f"Final YOLO YAML created: {OUTPUT_YAML}")
        print("STATUS: READY FOR TRAINING CONFIGURATION")
    else:
        print("STATUS: FIX ERRORS BEFORE CONTINUING")
        sys.exit(1)

if __name__ == "__main__":
    main()
