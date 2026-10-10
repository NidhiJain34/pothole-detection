from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train",
    "valid": ROOT / "valid",
    "test": ROOT / "test",
}

OUTPUT_ROOT = ROOT / "cleaned_labels"

EXPECTED_POLYGONS = {
    "train": 274,
    "valid": 61,
    "test": 24,
}


def polygon_to_box(parts):
    class_id = int(parts[0])
    coords = [float(value) for value in parts[1:]]

    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError("Invalid polygon coordinate count.")

    if not all(0 <= value <= 1 for value in coords):
        raise ValueError("Polygon coordinate outside [0, 1].")

    points = list(zip(coords[0::2], coords[1::2]))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError("Polygon has zero width or height.")

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    return (
        f"{class_id} "
        f"{x_center:.8f} "
        f"{y_center:.8f} "
        f"{width:.8f} "
        f"{height:.8f}"
    )


def main():
    if OUTPUT_ROOT.exists():
        raise RuntimeError(
            f"Output folder already exists:\n{OUTPUT_ROOT}\n"
            "Inspect or rename it before running again."
        )

    total_polygons = 0
    total_boxes = 0
    split_counts = {}

    try:
        for split, split_root in SPLITS.items():
            source = split_root / "labels"
            destination = OUTPUT_ROOT / split / "labels"
            destination.mkdir(parents=True, exist_ok=True)

            converted = 0

            for label_path in sorted(source.glob("*.txt")):
                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()

                new_lines = []

                for line in lines:
                    if not line.strip():
                        continue

                    parts = line.split()

                    # Standard YOLO detection annotation.
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        coords = [float(v) for v in parts[1:]]

                        if class_id != 0:
                            raise ValueError(
                                f"Unexpected class ID in {label_path.name}"
                            )

                        if not all(0 <= v <= 1 for v in coords):
                            raise ValueError(
                                f"Invalid box coordinates in {label_path.name}"
                            )

                        if coords[2] <= 0 or coords[3] <= 0:
                            raise ValueError(
                                f"Invalid box dimensions in {label_path.name}"
                            )

                        new_lines.append(line)
                    else:
                        # Convert a segmentation polygon to a box.
                        new_lines.append(polygon_to_box(parts))
                        converted += 1

                output_file = destination / label_path.name
                output_file.write_text(
                    "\n".join(new_lines) + ("\n" if new_lines else ""),
                    encoding="utf-8"
                )

            split_counts[split] = converted
            total_polygons += converted

            print(f"{split.upper()}: converted {converted} polygons")

        print("\n" + "=" * 50)
        print("CONVERSION SUMMARY")
        print("=" * 50)

        for split, count in split_counts.items():
            expected = EXPECTED_POLYGONS[split]
            status = "PASS" if count == expected else "CHECK"
            print(f"{split}: {count} polygons converted — {status}")

        print(f"\nTotal polygons converted: {total_polygons}")
        print(f"Output folder: {OUTPUT_ROOT}")

        if total_polygons != sum(EXPECTED_POLYGONS.values()):
            raise RuntimeError(
                "Converted polygon count differs from expectations."
            )

        print("\nSUCCESS: Converted labels saved separately.")
        print("Original labels were not modified.")

    except Exception:
        print("\nConversion stopped. Inspect the output folder.")
        raise


if __name__ == "__main__":
    main()