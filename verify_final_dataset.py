from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "TRAIN": ROOT / "train",
    "VALID": ROOT / "valid",
    "TEST": ROOT / "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def verify_split(name, folder):
    images_folder = folder / "images"
    labels_folder = folder / "labels"

    if not images_folder.is_dir() or not labels_folder.is_dir():
        print(f"\n{name}: ERROR - images or labels folder missing.")
        return None

    images = [
        p for p in images_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    annotated = 0
    backgrounds = 0
    missing_labels = []
    unreadable_labels = []

    for image in images:
        label = labels_folder / f"{image.stem}.txt"

        if not label.is_file():
            missing_labels.append(image.name)
            continue

        try:
            content = label.read_text(encoding="utf-8").strip()
        except Exception:
            unreadable_labels.append(label.name)
            continue

        if content:
            annotated += 1
        else:
            backgrounds += 1

    print(f"\n{name}")
    print("-" * 40)
    print(f"Total images:           {len(images)}")
    print(f"Pothole-annotated:      {annotated}")
    print(f"Background images:      {backgrounds}")
    print(f"Missing labels:         {len(missing_labels)}")
    print(f"Unreadable labels:      {len(unreadable_labels)}")

    if missing_labels:
        print("Examples of missing labels:", missing_labels[:5])

    if unreadable_labels:
        print("Examples of unreadable labels:", unreadable_labels[:5])

    return {
        "total": len(images),
        "annotated": annotated,
        "backgrounds": backgrounds,
        "missing": len(missing_labels),
        "unreadable": len(unreadable_labels),
    }


def main():
    print("=" * 50)
    print("FINAL DATASET VERIFICATION")
    print("=" * 50)

    results = {}

    for name, folder in SPLITS.items():
        results[name] = verify_split(name, folder)

    if any(value is None for value in results.values()):
        print("\nVERIFICATION FAILED: A dataset split is missing.")
        return

    print("\n" + "=" * 50)
    print("EXPECTED COUNTS")
    print("=" * 50)

    expected = {
        "TRAIN": (22832, 11416, 11416),
        "VALID": (3274, 2395, 879),
        "TEST": (2252, 1277, 975),
    }

    all_passed = True

    for name, (total, annotated, backgrounds) in expected.items():
        result = results[name]

        passed = (
            result["total"] == total
            and result["annotated"] == annotated
            and result["backgrounds"] == backgrounds
            and result["missing"] == 0
            and result["unreadable"] == 0
        )

        print(f"{name}: {'PASS' if passed else 'CHECK REQUIRED'}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("SUCCESS: All dataset counts match expectations.")
        print("Validation and test counts remain unchanged.")
    else:
        print("Dataset counts differ from expectations.")
        print("Review the results before starting training.")


if __name__ == "__main__":
    main()