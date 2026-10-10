from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "deduplicated_dataset"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("=" * 60)
    print("FINAL DATASET CLEANING CHECK")
    print("=" * 60)

    if not DATASET.is_dir():
        print(f"ERROR: Dataset not found: {DATASET}")
        return

    all_hashes = defaultdict(list)
    total_images = 0
    total_boxes = 0
    total_empty = 0
    total_errors = 0
    total_missing = 0
    total_orphan = 0

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            print(f"ERROR: Missing folders for {split}")
            return

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        label_files = {
            p.stem: p for p in label_dir.glob("*.txt")
        }

        image_stems = {p.stem for p in images}
        missing = image_stems - label_files.keys()
        orphan = label_files.keys() - image_stems

        boxes = 0
        empty = 0
        errors = 0

        for image in images:
            all_hashes[image_hash(image)].append((split, image.name))

            label_path = label_files.get(image.stem)

            if label_path is None:
                continue

            try:
                lines = [
                    line.strip()
                    for line in label_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
            except OSError:
                errors += 1
                continue

            if not lines:
                empty += 1
                continue

            for line in lines:
                values = line.split()

                if len(values) != 5:
                    errors += 1
                    continue

                try:
                    class_id = int(values[0])
                    x, y, w, h = map(float, values[1:])
                except ValueError:
                    errors += 1
                    continue

                if (
                    class_id != 0
                    or not all(0 <= v <= 1 for v in (x, y, w, h))
                    or w <= 0
                    or h <= 0
                    or x - w / 2 < -1e-6
                    or y - h / 2 < -1e-6
                    or x + w / 2 > 1 + 1e-6
                    or y + h / 2 > 1 + 1e-6
                ):
                    errors += 1
                else:
                    boxes += 1

        total_images += len(images)
        total_boxes += boxes
        total_empty += empty
        total_errors += errors
        total_missing += len(missing)
        total_orphan += len(orphan)

        print(f"\n{split.upper()}")
        print("-" * 40)
        print(f"Images: {len(images)}")
        print(f"Label files: {len(label_files)}")
        print(f"Missing labels: {len(missing)}")
        print(f"Orphan labels: {len(orphan)}")
        print(f"Valid bounding boxes: {boxes}")
        print(f"Empty labels: {empty}")
        print(f"Invalid annotation lines: {errors}")

    duplicates = {
        digest: items
        for digest, items in all_hashes.items()
        if len(items) > 1
    }

    cross_split_duplicates = {
        digest: items
        for digest, items in duplicates.items()
        if len({split for split, _ in items}) > 1
    }

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Valid bounding boxes: {total_boxes}")
    print(f"Empty labels: {total_empty}")
    print(f"Missing labels: {total_missing}")
    print(f"Orphan labels: {total_orphan}")
    print(f"Invalid annotation lines: {total_errors}")
    print(f"Duplicate image groups: {len(duplicates)}")
    print(f"Cross-split duplicate groups: {len(cross_split_duplicates)}")

    report = ROOT / "final_cleaning_report.txt"

    with report.open("w", encoding="utf-8") as file:
        file.write("FINAL DATASET CLEANING REPORT\n\n")
        file.write(f"Images: {total_images}\n")
        file.write(f"Bounding boxes: {total_boxes}\n")
        file.write(f"Empty labels: {total_empty}\n")
        file.write(f"Missing labels: {total_missing}\n")
        file.write(f"Orphan labels: {total_orphan}\n")
        file.write(f"Invalid annotation lines: {total_errors}\n")
        file.write(f"Duplicate groups: {len(duplicates)}\n")
        file.write(
            f"Cross-split duplicate groups: "
            f"{len(cross_split_duplicates)}\n\n"
        )

        for digest, items in cross_split_duplicates.items():
            file.write(f"Duplicate hash: {digest}\n")
            for split, name in items:
                file.write(f"  {split}: {name}\n")
            file.write("\n")

    print(f"\nReport saved to: {report}")
    print("No images or labels were modified.")

    if (
        total_missing == 0
        and total_orphan == 0
        and total_errors == 0
        and len(cross_split_duplicates) == 0
    ):
        print("\nPASS: Structural cleaning checks passed.")
    else:
        print("\nREVIEW REQUIRED: Some cleaning checks need attention.")


if __name__ == "__main__":
    main()
    