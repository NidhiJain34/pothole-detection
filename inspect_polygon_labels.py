from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train" / "labels",
    "valid": ROOT / "valid" / "labels",
    "test": ROOT / "test" / "labels",
}


def inspect_polygon(line):
    parts = line.split()

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except (ValueError, IndexError):
        return False, "Invalid numeric format"

    if class_id != 0:
        return False, "Unexpected class ID"

    if len(coords) < 6:
        return False, "Too few polygon coordinates"

    if len(coords) % 2 != 0:
        return False, "Odd number of coordinates"

    if not all(0 <= value <= 1 for value in coords):
        return False, "Coordinate outside [0, 1]"

    points = list(zip(coords[0::2], coords[1::2]))

    if len(set(points)) < 3:
        return False, "Fewer than three distinct points"

    return True, "Valid polygon structure"


def main():
    report_path = ROOT / "polygon_inspection_report.txt"

    total_polygons = 0
    valid_polygons = 0
    invalid_polygons = 0

    with report_path.open("w", encoding="utf-8") as report:
        for split, labels_dir in SPLITS.items():
            report.write(f"\n{'=' * 50}\n")
            report.write(f"SPLIT: {split.upper()}\n")
            report.write(f"{'=' * 50}\n")

            split_total = 0
            split_valid = 0
            split_invalid = 0

            for label_path in sorted(labels_dir.glob("*.txt")):
                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()

                for line_number, line in enumerate(lines, start=1):
                    parts = line.split()

                    # Only inspect non-standard five-value lines.
                    if len(parts) == 5 or not parts:
                        continue

                    total_polygons += 1
                    split_total += 1

                    valid, reason = inspect_polygon(line)

                    if valid:
                        valid_polygons += 1
                        split_valid += 1
                    else:
                        invalid_polygons += 1
                        split_invalid += 1

                    report.write(
                        f"{label_path.name} | "
                        f"line {line_number} | "
                        f"{'VALID' if valid else 'INVALID'} | "
                        f"{reason}\n"
                    )

            report.write(f"Polygons found: {split_total}\n")
            report.write(f"Valid structures: {split_valid}\n")
            report.write(f"Invalid structures: {split_invalid}\n")

    print("POLYGON INSPECTION COMPLETE")
    print(f"Total polygons: {total_polygons}")
    print(f"Valid polygon structures: {valid_polygons}")
    print(f"Invalid polygon structures: {invalid_polygons}")
    print(f"Report saved to: {report_path}")
    print("No dataset labels were modified.")


if __name__ == "__main__":
    main()