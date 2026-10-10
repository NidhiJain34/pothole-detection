from pathlib import Path
import hashlib
from collections import defaultdict
import csv

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train" / "images",
    "valid": ROOT / "valid" / "images",
    "test": ROOT / "test" / "images",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calculate_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_label_path(image_path):
    """
    Convert:
    train/images/example.jpg
    into:
    train/labels/example.txt
    """
    return (
        image_path.parent.parent
        / "labels"
        / f"{image_path.stem}.txt"
    )


hash_groups = defaultdict(list)

print("Analyzing exact duplicate images...")

for split, image_dir in SPLITS.items():
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            file_hash = calculate_hash(image_path)
            hash_groups[file_hash].append((split, image_path))

duplicate_groups = [
    paths for paths in hash_groups.values()
    if len(paths) > 1
]

report_path = ROOT / "duplicate_analysis.csv"

with report_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow([
        "duplicate_group",
        "duplicate_type",
        "split",
        "image_path",
        "label_path",
        "label_exists",
    ])

    for group_number, group in enumerate(duplicate_groups, start=1):
        splits = sorted({split for split, _ in group})

        if len(splits) == 1:
            duplicate_type = f"Within {splits[0]}"
        else:
            duplicate_type = "Cross-split: " + " + ".join(splits)

        for split, image_path in group:
            label_path = get_label_path(image_path)

            writer.writerow([
                group_number,
                duplicate_type,
                split,
                str(image_path),
                str(label_path),
                label_path.exists(),
            ])

print("\n========== DUPLICATE ANALYSIS ==========")
print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Report saved to: {report_path}")

cross_split_groups = 0
within_train_groups = 0

for group in duplicate_groups:
    splits = {split for split, _ in group}

    if len(splits) > 1:
        cross_split_groups += 1

    if splits == {"train"}:
        within_train_groups += 1

print(f"Cross-split duplicate groups: {cross_split_groups}")
print(f"Train-only duplicate groups: {within_train_groups}")

print("\nFirst 10 duplicate groups:")

for index, group in enumerate(duplicate_groups[:10], start=1):
    print(f"\nGroup {index}:")

    for split, image_path in group:
        label_path = get_label_path(image_path)

        print(f"  Split: {split}")
        print(f"  Image: {image_path.name}")
        print(f"  Label exists: {label_path.exists()}")

print("\nNo files were deleted or modified.")