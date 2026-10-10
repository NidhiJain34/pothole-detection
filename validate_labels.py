from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train",
    "valid": ROOT / "valid",
    "test": ROOT / "test",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

EXPECTED_CLASS_ID = 0


def validate_split(split_name, split_path):
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images or labels folder for {split_name}"
        )

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_files = list(labels_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_files}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    malformed = []
    wrong_classes = []
    invalid_boxes = []
    unreadable_labels = []

    annotated_images = 0
    background_images = 0
    total_boxes = 0

    for stem in sorted(image_stems & label_stems):
        label_path = labels_dir / f"{stem}.txt"

        try:
            content = label_path.read_text(encoding="utf-8").strip()
        except Exception as error:
            unreadable_labels.append(
                (label_path.name, str(error))
            )
            continue

        if not content:
            background_images += 1
            continue

        annotated_images += 1

        for line_number, line in enumerate(
            content.splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                malformed.append(
                    (label_path.name, line_number, line)
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
            except ValueError:
                malformed.append(
                    (label_path.name, line_number, line)
                )
                continue

            total_boxes += 1

            if class_id != EXPECTED_CLASS_ID:
                wrong_classes.append(
                    (label_path.name, line_number, class_id)
                )

            values = [x, y, width, height]

            if not all(
                value == value
                and abs(value) != float("inf")
                for value in values
            ):
                invalid_boxes.append(
                    (label_path.name, line_number, values)
                )
                continue

            if not all(0 <= value <= 1 for value in values):
                invalid_boxes.append(
                    (label_path.name, line_number, values)
                )
                continue

            if width <= 0 or height <= 0:
                invalid_boxes.append(
                    (label_path.name, line_number, values)
                )
                continue

            if (
                x - width / 2 < -1e-6
                or y - height / 2 < -1e-6
                or x + width / 2 > 1 + 1e-6
                or y + height / 2 > 1 + 1e-6
            ):
                invalid_boxes.append(
                    (label_path.name, line_number, values)
                )

    print(f"\n{'=' * 55}")
    print(f"SPLIT: {split_name.upper()}")
    print("=" * 55)

    print(f"Images:               {len(images)}")
    print(f"Label files:          {len(label_files)}")
    print(f"Annotated images:     {annotated_images}")
    print(f"Background images:    {background_images}")
    print(f"Total bounding boxes: {total_boxes}")

    print(f"Missing labels:       {len(missing_labels)}")
    print(f"Orphan labels:        {len(orphan_labels)}")
    print(f"Malformed lines:      {len(malformed)}")
    print(f"Wrong class IDs:      {len(wrong_classes)}")
    print(f"Invalid boxes:        {len(invalid_boxes)}")
    print(f"Unreadable labels:    {len(unreadable_labels)}")

    issues = [
        ("Missing labels", sorted(missing_labels)),
        ("Orphan labels", sorted(orphan_labels)),
        ("Malformed lines", malformed),
        ("Wrong class IDs", wrong_classes),
        ("Invalid boxes", invalid_boxes),
        ("Unreadable labels", unreadable_labels),
    ]

    for title, entries in issues:
        if entries:
            print(f"\nExamples — {title}:")
            for entry in list(entries)[:5]:
                print(" ", entry)

    return (
        len(missing_labels)
        + len(orphan_labels)
        + len(malformed)
        + len(wrong_classes)
        + len(invalid_boxes)
        + len(unreadable_labels)
    )


def main():
    print("YOLO LABEL VALIDATION")
    print("Read-only check: no files will be modified.")

    total_issues = 0

    for name, folder in SPLITS.items():
        total_issues += validate_split(name, folder)

    print(f"\n{'=' * 55}")
    print("OVERALL RESULT")
    print("=" * 55)

    if total_issues == 0:
        print("PASS: No structural label issues detected.")
    else:
        print(f"CHECK REQUIRED: {total_issues} issues detected.")
        print("Review the reported examples before making changes.")


if __name__ == "__main__":
    main()