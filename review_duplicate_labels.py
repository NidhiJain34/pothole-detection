from pathlib import Path
from collections import defaultdict
import hashlib
import csv

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train" / "images",
    "valid": ROOT / "valid" / "images",
    "test": ROOT / "test" / "images",
}

CLEANED_LABELS = ROOT / "cleaned_labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_cleaned_label(split, image_path):
    return (
        CLEANED_LABELS
        / split
        / "labels"
        / f"{image_path.stem}.txt"
    )


def read_label(label_path):
    if not label_path.exists():
        return None

    # Ignore harmless whitespace differences.
    return sorted(
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


# Group images by exact file content.
groups = defaultdict(list)

print("Scanning images and comparing their cleaned labels...")

for split, image_dir in SPLITS.items():
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            groups[file_hash(image_path)].append((split, image_path))

duplicates = [
    group for group in groups.values()
    if len(group) > 1
]

report_path = ROOT / "duplicate_label_review.csv"

cross_split_count = 0
matching_labels_count = 0
different_labels_count = 0
missing_labels_count = 0

with report_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "group",
        "duplicate_type",
        "label_comparison",
        "split",
        "image",
        "cleaned_label",
        "label_exists",
        "suggested_action",
    ])

    for number, group in enumerate(duplicates, start=1):
        splits = {split for split, _ in group}
        labels = []

        for split, image_path in group:
            label_path = get_cleaned_label(split, image_path)
            labels.append(read_label(label_path))

        if len(splits) > 1:
            duplicate_type = "CROSS-SPLIT"
            cross_split_count += 1
        else:
            duplicate_type = "WITHIN-SPLIT"

        if any(label is None for label in labels):
            comparison = "MISSING LABEL"
            missing_labels_count += 1
        elif all(label == labels[0] for label in labels):
            comparison = "LABELS MATCH"
            matching_labels_count += 1
        else:
            comparison = "LABELS DIFFER"
            different_labels_count += 1

        for split, image_path in group:
            label_path = get_cleaned_label(split, image_path)

            if comparison == "MISSING LABEL":
                action = "MANUAL REVIEW"
            elif comparison == "LABELS DIFFER":
                action = "MANUAL REVIEW"
            elif duplicate_type == "CROSS-SPLIT":
                action = (
                    "KEEP VALID/TEST COPY; REVIEW TRAIN COPY"
                    if split == "train"
                    else "KEEP EVALUATION COPY"
                )
            else:
                action = "REVIEW FOR REDUNDANCY"

            writer.writerow([
                number,
                duplicate_type,
                comparison,
                split,
                str(image_path),
                str(label_path),
                label_path.exists(),
                action,
            ])

print("\n========== REVIEW RESULTS ==========")
print(f"Duplicate groups: {len(duplicates)}")
print(f"Cross-split groups: {cross_split_count}")
print(f"Groups with matching labels: {matching_labels_count}")
print(f"Groups with different labels: {different_labels_count}")
print(f"Groups with missing labels: {missing_labels_count}")
print(f"\nReport saved to: {report_path}")
print("\nNo images or labels were deleted or modified.")