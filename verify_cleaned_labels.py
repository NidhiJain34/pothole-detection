from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent
CLEANED_ROOT = ROOT / "cleaned_labels"

SPLITS = ["train", "valid", "test"]


def main():
    total_files = 0
    total_boxes = 0
    total_backgrounds = 0
    total_errors = 0

    print("=" * 55)
    print("VERIFYING CONVERTED YOLO LABELS")
    print("=" * 55)

    for split in SPLITS:
        label_dir = CLEANED_ROOT / split / "labels"

        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing directory: {label_dir}"
            )

        files = list(label_dir.glob("*.txt"))
        boxes = 0
        backgrounds = 0
        errors = []

        for label_file in files:
            content = label_file.read_text(
                encoding="utf-8"
            ).strip()

            if not content:
                backgrounds += 1
                continue

            for line_number, line in enumerate(
                content.splitlines(), start=1
            ):
                parts = line.split()

                if len(parts) != 5:
                    errors.append(
                        (label_file.name, line_number,
                         "Expected 5 values")
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, width, height = map(
                        float, parts[1:]
                    )
                except ValueError:
                    errors.append(
                        (label_file.name, line_number,
                         "Invalid numeric values")
                    )
                    continue

                values = [x, y, width, height]

                if class_id != 0:
                    errors.append(
                        (label_file.name, line_number,
                         "Unexpected class ID")
                    )
                    continue

                if not all(math.isfinite(v) for v in values):
                    errors.append(
                        (label_file.name, line_number,
                         "Non-finite coordinate")
                    )
                    continue

                if not all(0 <= v <= 1 for v in values):
                    errors.append(
                        (label_file.name, line_number,
                         "Coordinate outside [0, 1]")
                    )
                    continue

                if width <= 0 or height <= 0:
                    errors.append(
                        (label_file.name, line_number,
                         "Non-positive box dimensions")
                    )
                    continue

                if (
                    x - width / 2 < -1e-6
                    or y - height / 2 < -1e-6
                    or x + width / 2 > 1 + 1e-6
                    or y + height / 2 > 1 + 1e-6
                ):
                    errors.append(
                        (label_file.name, line_number,
                         "Box extends outside image")
                    )
                    continue

                boxes += 1

        print(f"\n{split.upper()}")
        print("-" * 40)
        print(f"Label files: {len(files)}")
        print(f"Valid boxes: {boxes}")
        print(f"Empty labels: {backgrounds}")
        print(f"Validation errors: {len(errors)}")

        for error in errors[:5]:
            print(" ", error)

        total_files += len(files)
        total_boxes += boxes
        total_backgrounds += backgrounds
        total_errors += len(errors)

    print("\n" + "=" * 55)
    print("OVERALL RESULTS")
    print("=" * 55)
    print(f"Label files: {total_files}")
    print(f"Valid boxes: {total_boxes}")
    print(f"Empty labels: {total_backgrounds}")
    print(f"Validation errors: {total_errors}")

    if total_errors == 0:
        print("\nPASS: All converted labels are structurally valid.")
    else:
        print("\nCHECK REQUIRED: Review the reported errors.")

    print("Original dataset labels were not modified.")


if __name__ == "__main__":
    main()