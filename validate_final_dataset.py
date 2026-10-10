from pathlib import Path
from collections import Counter, defaultdict
import math
import sys

try:
    from PIL import Image, ImageFile
except ImportError:
    sys.exit(
        "Pillow is required for image-readability checks.\n"
        "Install it in your project venv with:\n"
        ".venv/bin/python -m pip install Pillow"
    )

ROOT = Path("/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26")
DATASET = ROOT / "final_cleaned_dataset_no_conflicts"
REPORT = ROOT / "final_dataset_validation_report.txt"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ImageFile.LOAD_TRUNCATED_IMAGES = False

def is_number(value):
    try:
        x = float(value)
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False

def main():
    if not DATASET.is_dir():
        sys.exit(f"ERROR: Dataset folder not found:\n{DATASET}")

    lines = [
        "FINAL DATASET VALIDATION REPORT",
        "=" * 60,
        f"Dataset: {DATASET}",
        "Expected YOLO format: class x_center y_center width height",
        ""
    ]
    total = Counter()
    errors = []
    warnings = []
    hashes_not_checked = True  # Duplicate removal already verified by previous script.

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing images or labels directory")
            continue

        images = sorted(p for p in image_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)

        split_count = Counter()
        bad_images = []
        invalid_label_files = []
        invalid_lines = []
        boxes = 0
        empty_labels = 0
        class_counts = Counter()

        for img in images:
            split_count["images"] += 1
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    width, height = im.size
                    im.load()
                    if width <= 0 or height <= 0:
                        raise ValueError("non-positive dimensions")
            except Exception as exc:
                bad_images.append(f"{img.name}: {exc}")

            label_path = label_dir / f"{img.stem}.txt"
            if not label_path.is_file():
                continue

            try:
                raw = label_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                invalid_label_files.append(f"{label_path.name}: cannot read ({exc})")
                continue

            if not raw:
                empty_labels += 1
                continue

            for line_no, row in enumerate(raw.splitlines(), start=1):
                parts = row.split()
                if len(parts) != 5:
                    invalid_lines.append(f"{label_path.name}:{line_no}: expected 5 values, got {len(parts)}")
                    continue
                if not parts[0].isdigit():
                    invalid_lines.append(f"{label_path.name}:{line_no}: class ID is not a non-negative integer")
                    continue
                values = parts[1:]
                if not all(is_number(v) for v in values):
                    invalid_lines.append(f"{label_path.name}:{line_no}: non-numeric or non-finite coordinate")
                    continue
                x, y, w, h = map(float, values)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid_lines.append(f"{label_path.name}:{line_no}: coordinates/dimensions outside valid YOLO range")
                    continue
                if x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
                    invalid_lines.append(f"{label_path.name}:{line_no}: bounding box extends outside image")
                    continue
                class_id = int(parts[0])
                class_counts[class_id] += 1
                boxes += 1

        backgrounds = empty_labels
        annotated = split_count["images"] - len(missing_labels) - backgrounds
        total["images"] += len(images)
        total["labels"] += len(labels)
        total["missing_labels"] += len(missing_labels)
        total["orphan_labels"] += len(orphan_labels)
        total["empty_labels"] += backgrounds
        total["boxes"] += boxes
        total["bad_images"] += len(bad_images)
        total["invalid_lines"] += len(invalid_lines)
        total["invalid_label_files"] += len(invalid_label_files)

        lines.extend([
            f"{split.upper()}",
            "-" * 40,
            f"Images: {len(images)}",
            f"Label files: {len(labels)}",
            f"Images missing labels: {len(missing_labels)}",
            f"Orphan label files: {len(orphan_labels)}",
            f"Annotated images (estimated): {annotated}",
            f"Empty-label/background images: {backgrounds}",
            f"Valid bounding boxes counted: {boxes}",
            f"Unreadable/corrupt images: {len(bad_images)}",
            f"Invalid annotation lines: {len(invalid_lines)}",
            f"Unreadable label files: {len(invalid_label_files)}",
            f"Class IDs found: {dict(sorted(class_counts.items()))}",
            ""
        ])

        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} images have no matching label file")
            lines.append("First missing-label examples: " + ", ".join(missing_labels[:10]))
        if orphan_labels:
            errors.append(f"{split}: {len(orphan_labels)} orphan label files")
            lines.append("First orphan-label examples: " + ", ".join(orphan_labels[:10]))
        if bad_images:
            errors.append(f"{split}: {len(bad_images)} unreadable images")
            lines.append("First unreadable images: " + " | ".join(bad_images[:5]))
        if invalid_lines:
            errors.append(f"{split}: {len(invalid_lines)} invalid annotation lines")
            lines.append("First invalid lines: " + " | ".join(invalid_lines[:10]))
        if invalid_label_files:
            errors.append(f"{split}: {len(invalid_label_files)} unreadable label files")
            lines.append("Unreadable label examples: " + " | ".join(invalid_label_files[:5]))
        if class_counts and any(cid != 0 for cid in class_counts):
            warnings.append(f"{split}: nonzero class IDs found; expected only class 0 for pothole")
        if len(images) != len(labels):
            warnings.append(f"{split}: image count and label-file count differ")

        lines.append("")

    lines.extend([
        "OVERALL SUMMARY",
        "=" * 60,
        f"Total images: {total['images']}",
        f"Total label files: {total['labels']}",
        f"Total empty labels/backgrounds: {total['empty_labels']}",
        f"Total valid boxes counted: {total['boxes']}",
        f"Images missing labels: {total['missing_labels']}",
        f"Orphan label files: {total['orphan_labels']}",
        f"Unreadable images: {total['bad_images']}",
        f"Invalid annotation lines: {total['invalid_lines']}",
        f"Unreadable label files: {total['invalid_label_files']}",
        "",
        "ERRORS:",
        *(errors if errors else ["None detected by these checks."]),
        "",
        "WARNINGS:",
        *(warnings if warnings else ["None detected by these checks."]),
        "",
        "NOTE: Empty label files are treated as background images. This script checks file/label structure and coordinate validity, not whether each box accurately surrounds a pothole.",
    ])

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FINAL DATASET VALIDATION FINISHED")
    print(f"Dataset: {DATASET}")
    print(f"Total images: {total['images']}")
    print(f"Total label files: {total['labels']}")
    print(f"Background images (empty labels): {total['empty_labels']}")
    print(f"Valid boxes counted: {total['boxes']}")
    print(f"Missing labels: {total['missing_labels']}")
    print(f"Orphan labels: {total['orphan_labels']}")
    print(f"Unreadable images: {total['bad_images']}")
    print(f"Invalid annotation lines: {total['invalid_lines']}")
    print(f"Unreadable label files: {total['invalid_label_files']}")
    print(f"\nReport saved to: {REPORT}")
    print("\nSTATUS:", "CHECK REQUIRED — see report" if errors else "STRUCTURAL CHECKS PASSED")
    if warnings:
        print("Warnings found; see report.")
    print("Note: this does not prove that bounding boxes visually match potholes.")

if __name__ == "__main__":
    main()
