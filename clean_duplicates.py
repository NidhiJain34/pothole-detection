from pathlib import Path
from collections import defaultdict
import hashlib
import shutil
import csv

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train" / "images",
    "valid": ROOT / "valid" / "images",
    "test": ROOT / "test" / "images",
}

CLEANED_LABELS = ROOT / "cleaned_labels"

OUTPUT = ROOT / "deduplicated_dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calculate_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_label(split, image):
    path = CLEANED_LABELS / split / "labels" / f"{image.stem}.txt"

    if not path.exists():
        return None

    return sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main():
    if OUTPUT.exists():
        print(f"ERROR: Output folder already exists:\n{OUTPUT}")
        print("Rename the existing folder before running again.")
        return

    print("Scanning images and identifying duplicates...")

    groups = defaultdict(list)

    for split, image_dir in SPLITS.items():
        if not image_dir.is_dir():
            print(f"ERROR: Image folder missing: {image_dir}")
            return

        label_dir = CLEANED_LABELS / split / "labels"

        if not label_dir.is_dir():
            print(f"ERROR: Cleaned label folder missing: {label_dir}")
            return

        for image in image_dir.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                groups[calculate_hash(image)].append((split, image))

    # Check all labels before creating the output.
    for group in groups.values():
        for split, image in group:
            label = get_label(split, image)

            if label is None:
                print(f"ERROR: Missing cleaned label for {split}/{image.name}")
                print("No dataset was created. Check the cleaned labels.")
                return

    selected = []
    report_rows = []
    conflict_groups = 0
    duplicate_groups = 0

    print("Reviewing duplicate groups...")

    for group_number, group in enumerate(groups.values(), start=1):
        if len(group) == 1:
            selected.append(group[0])
            continue

        duplicate_groups += 1

        # Compare labels from the validated cleaned-label folders.
        label_contents = [
            get_label(split, image)
            for split, image in group
        ]

        labels_match = all(
            labels == label_contents[0]
            for labels in label_contents
        )

        # Prefer test, then validation, then training.
        priority = {"test": 0, "valid": 1, "train": 2}
        group_sorted = sorted(
            group,
            key=lambda item: (priority[item[0]], item[1].name)
        )

        chosen = group_sorted[0]
        selected.append(chosen)

        if not labels_match:
            conflict_groups += 1
            status = "LABEL CONFLICT - REVIEW"
        else:
            status = "LABELS MATCH"

        for split, image in group:
            if (split, image) == chosen:
                action = "KEEP"
            else:
                action = "OMIT DUPLICATE"

            report_rows.append([
                group_number,
                status,
                split,
                image.name,
                action,
                chosen[0],
                chosen[1].name,
            ])

    print("Creating the separate deduplicated dataset...")

    # Create a new dataset without modifying the original.
    for split in SPLITS:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

    for split, image in selected:
        output_image = OUTPUT / split / "images" / image.name
        output_label = OUTPUT / split / "labels" / f"{image.stem}.txt"

        source_label = (
            CLEANED_LABELS / split / "labels" / f"{image.stem}.txt"
        )

        shutil.copy2(image, output_image)
        shutil.copy2(source_label, output_label)

    report_path = OUTPUT / "duplicate_cleanup_report.csv"

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "group",
            "label_status",
            "original_split",
            "image",
            "action",
            "kept_split",
            "kept_image",
        ])
        writer.writerows(report_rows)

    print("\n" + "=" * 45)
    print("DEDUPLICATION COMPLETE")
    print("=" * 45)

    total = 0

    for split in SPLITS:
        count = sum(
            1 for path in (OUTPUT / split / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        total += count
        print(f"{split.upper()}: {count} images")

    print(f"\nTotal images in new dataset: {total}")
    print(f"Duplicate groups processed: {duplicate_groups}")
    print(f"Groups with conflicting labels: {conflict_groups}")
    print(f"Review report: {report_path}")
    print(f"Output folder: {OUTPUT}")
    print("\nOriginal images and labels remain unchanged.")
    print("Conflicting label groups are recorded for review.")


if __name__ == "__main__":
    main()