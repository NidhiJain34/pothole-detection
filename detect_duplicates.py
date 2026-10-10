from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parent

SPLITS = {
    "train": ROOT / "train" / "images",
    "valid": ROOT / "valid" / "images",
    "test": ROOT / "test" / "images",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def calculate_hash(path):
    """Calculate an exact file-content hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    hash_groups = defaultdict(list)
    total_images = 0

    print("=" * 60)
    print("EXACT DUPLICATE IMAGE DETECTION")
    print("=" * 60)

    for split, folder in SPLITS.items():
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Images folder not found: {folder}"
            )

        images = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{split.upper()}: {len(images)} images")

        for image in images:
            total_images += 1
            digest = calculate_hash(image)
            hash_groups[digest].append((split, image))

    duplicate_groups = [
        group for group in hash_groups.values()
        if len(group) > 1
    ]

    duplicate_file_count = sum(
        len(group) - 1 for group in duplicate_groups
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Total images checked: {total_images}")
    print(f"Unique file contents: {len(hash_groups)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Extra duplicate files: {duplicate_file_count}")

    for index, group in enumerate(duplicate_groups[:20], start=1):
        print(f"\nDuplicate group {index}:")

        for split, path in group:
            print(f"  [{split}] {path.name}")

    report_path = ROOT / "duplicate_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"Total images: {total_images}\n")
        report.write(f"Unique contents: {len(hash_groups)}\n")
        report.write(f"Duplicate groups: {len(duplicate_groups)}\n")
        report.write(f"Extra duplicate files: {duplicate_file_count}\n")

        for index, group in enumerate(duplicate_groups, start=1):
            report.write(f"\nDuplicate group {index}:\n")

            for split, path in group:
                report.write(f"[{split}] {path}\n")

    print(f"\nFull report saved to: {report_path}")
    print("No images were deleted or modified.")


if __name__ == "__main__":
    main()