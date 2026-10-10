from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train" / "labels",
    "valid": ROOT / "valid" / "labels",
    "test": ROOT / "test" / "labels",
}


def diagnose_line(line):
    parts = line.split()

    if len(parts) < 5:
        return "Too few values"

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return "Non-numeric values"

    if class_id != 0:
        return f"Unexpected class ID: {class_id}"

    if not all(0 <= value <= 1 for value in coords):
        return "Coordinate outside [0, 1]"

    # Standard detection format:
    if len(parts) == 5:
        return "YOLO bounding box"

    # Polygon format: x/y coordinate pairs.
    if len(coords) >= 6 and len(coords) % 2 == 0:
        return "Possible YOLO segmentation polygon"

    return "Unknown format"


def main():
    totals = {}

    for split, labels_dir in SPLITS.items():
        counts = {}
        examples = {}

        for label_path in sorted(labels_dir.glob("*.txt")):
            try:
                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except Exception as error:
                print(f"Could not read {label_path}: {error}")
                continue

            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue

                result = diagnose_line(line)
                counts[result] = counts.get(result, 0) + 1

                if result not in examples:
                    examples[result] = (
                        label_path.name,
                        line_number,
                        line[:180]
                    )

        totals[split] = counts

        print(f"\n{'=' * 55}")
        print(f"SPLIT: {split.upper()}")
        print("=" * 55)

        for category, count in sorted(counts.items()):
            print(f"{category}: {count}")
            print(" Example:", examples[category])

    print(f"\n{'=' * 55}")
    print("DIAGNOSIS COMPLETE")
    print("=" * 55)
    print("No files were modified.")


if __name__ == "__main__":
    main()