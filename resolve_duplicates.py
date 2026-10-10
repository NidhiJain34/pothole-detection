from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "deduplicated_dataset"
OUTPUT = ROOT / "final_cleaned_dataset"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lower number = higher priority.
# Keep test copies first, then validation, then training.
PRIORITY = {
    "test": 0,
    "valid": 1,
    "train": 2,
}


def sha256_file(path):
    """Calculate a file's SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalized_label(label_path):
    """Read label content for comparison."""
    if not label_path.exists():
        return None

    return [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]


def main():
    if not SOURCE.is_dir():
        print(f"ERROR: Source dataset not found:\n{SOURCE}")
        return

    if OUTPUT.exists():
        print("ERROR: Output folder already exists:")
        print(OUTPUT)
        print(
            "\nNothing was overwritten. Rename or move the existing "
            "output folder before running this script again."
        )
        return

    # Collect every image and group identical files by hash.
    hash_groups = defaultdict(list)
    image_count = 0

    print("Scanning images and finding exact duplicates...")

    for split in SPLITS:
        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            print(f"ERROR: Missing image or label folder for {split}.")
            return

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image in images:
            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                print(f"ERROR: Missing label for {image}")
                print("No output dataset was created.")
                return

            digest = sha256_file(image)

            hash_groups[digest].append({
                "split": split,
                "image": image,
                "label": label,
            })

            image_count += 1

        print(f"{split}: scanned {len(images)} images")

    duplicate_groups = [
        group for group in hash_groups.values()
        if len(group) > 1
    ]

    # Select one copy from every group.
    # Test is preferred over validation, which is preferred over train.
    selected = []
    duplicate_log = []
    label_conflicts = []
    cross_split_groups = 0
    removed_copies = 0

    for group in hash_groups.values():
        group = sorted(
            group,
            key=lambda item: (
                PRIORITY[item["split"]],
                str(item["image"]).lower(),
            ),
        )

        winner = group[0]
        selected.append(winner)

        if len(group) == 1:
            continue

        removed_copies += len(group) - 1

        splits_present = sorted({item["split"] for item in group})

        if len(splits_present) > 1:
            cross_split_groups += 1

        label_versions = {
            tuple(normalized_label(item["label"]) or [])
            for item in group
        }

        if len(label_versions) > 1:
            label_conflicts.append(group)

        duplicate_log.append({
            "winner": winner,
            "copies": group,
            "splits": splits_present,
            "conflict": len(label_versions) > 1,
        })

    # Create a fresh dataset. Original folders remain unchanged.
    print("\nCreating final_cleaned_dataset...")

    for split in SPLITS:
        (OUTPUT / split / "images").mkdir(parents=True)
        (OUTPUT / split / "labels").mkdir(parents=True)

    for item in selected:
        split = item["split"]

        destination_image = OUTPUT / split / "images" / item["image"].name
        destination_label = OUTPUT / split / "labels" / item["label"].name

        shutil.copy2(item["image"], destination_image)
        shutil.copy2(item["label"], destination_label)

    # Write a report with every duplicate group and any label conflicts.
    report_path = ROOT / "duplicate_resolution_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("DUPLICATE RESOLUTION REPORT\n")
        report.write("=" * 65 + "\n\n")
        report.write(f"Source: {SOURCE}\n")
        report.write(f"Output: {OUTPUT}\n")
        report.write(f"Images scanned: {image_count}\n")
        report.write(f"Duplicate groups found: {len(duplicate_groups)}\n")
        report.write(f"Cross-split duplicate groups: {cross_split_groups}\n")
        report.write(f"Extra image copies removed: {removed_copies}\n")
        report.write(f"Groups with conflicting labels: {len(label_conflicts)}\n")
        report.write(f"Images retained: {len(selected)}\n\n")

        for number, entry in enumerate(duplicate_log, start=1):
            report.write(f"Duplicate group {number}\n")
            report.write(f"Splits involved: {', '.join(entry['splits'])}\n")
            report.write(f"Kept image: {entry['winner']['image']}\n")
            report.write(
                "Label conflict: "
                + ("YES - selected copy's label retained\n"
                   if entry["conflict"] else "NO\n")
            )

            for item in entry["copies"]:
                report.write(
                    f"  Copy: {item['image']} | "
                    f"Split: {item['split']}\n"
                )

            report.write("\n")

        report.write("\nLABEL CONFLICT DETAILS\n")
        report.write("=" * 65 + "\n")

        if not label_conflicts:
            report.write("No duplicate groups with differing labels.\n")
        else:
            report.write(
                "The selected image's label was retained automatically. "
                "These groups need targeted review because the labels differ.\n\n"
            )

            for number, group in enumerate(label_conflicts, start=1):
                report.write(f"Conflict {number}\n")

                for item in group:
                    report.write(
                        f"Image: {item['image']}\n"
                        f"Split: {item['split']}\n"
                        f"Label: {item['label']}\n"
                        f"Label contents: "
                        f"{normalized_label(item['label'])}\n\n"
                    )

    print("\n" + "=" * 55)
    print("DUPLICATE RESOLUTION COMPLETE")
    print("=" * 55)

    print(f"Images scanned: {image_count}")
    print(f"Duplicate groups found: {len(duplicate_groups)}")
    print(f"Cross-split duplicate groups: {cross_split_groups}")
    print(f"Extra copies removed: {removed_copies}")
    print(f"Images retained: {len(selected)}")
    print(f"Groups with conflicting labels: {len(label_conflicts)}")

    for split in SPLITS:
        image_dir = OUTPUT / split / "images"
        count = sum(
            1 for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"{split.upper()} images: {count}")

    print(f"\nNew dataset: {OUTPUT}")
    print(f"Detailed report: {report_path}")

    if label_conflicts:
        print("\nREVIEW NEEDED: Some identical images have different labels.")
        print("See LABEL CONFLICT DETAILS in the report.")
    else:
        print("\nNo conflicting labels were found among duplicate images.")


if __name__ == "__main__":
    main()