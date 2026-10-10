from pathlib import Path
from collections import defaultdict
import hashlib
import shutil
import sys

ROOT = Path("/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26")
SOURCE = ROOT / "deduplicated_dataset"
OUTPUT = ROOT / "final_cleaned_dataset_no_conflicts"
REPORT = ROOT / "remove_conflicting_duplicates_report.txt"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not SOURCE.is_dir():
        sys.exit(f"ERROR: Source dataset not found: {SOURCE}")
    if OUTPUT.exists():
        sys.exit(
            f"ERROR: Output folder already exists:\n{OUTPUT}\n"
            "Nothing was changed. Rename or remove that output folder manually, "
            "then run this script again."
        )

    # Index images from all splits by exact file content.
    by_hash = defaultdict(list)
    for split in SPLITS:
        image_dir = SOURCE / split / "images"
        if not image_dir.is_dir():
            sys.exit(f"ERROR: Missing image folder: {image_dir}")
        for img in sorted(image_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                by_hash[sha256(img)].append((split, img))

    duplicate_groups = {digest: items for digest, items in by_hash.items() if len(items) > 1}
    duplicate_images = sum(len(items) for items in duplicate_groups.values())

    # As requested, remove ALL copies belonging to duplicate groups.
    # Keep every unique image and its corresponding label.
    kept = {split: 0 for split in SPLITS}
    removed_by_split = {split: 0 for split in SPLITS}
    missing_labels = []
    copied = []

    try:
        for split in SPLITS:
            (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
            (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

        for digest, items in by_hash.items():
            if len(items) > 1:
                for split, img in items:
                    removed_by_split[split] += 1
                continue

            split, img = items[0]
            label = SOURCE / split / "labels" / (img.stem + ".txt")
            if not label.is_file():
                missing_labels.append(str(label))
                continue

            dest_img = OUTPUT / split / "images" / img.name
            dest_label = OUTPUT / split / "labels" / label.name
            shutil.copy2(img, dest_img)
            shutil.copy2(label, dest_label)
            kept[split] += 1
            copied.append((dest_img, dest_label))

        # Stop rather than leave a seemingly valid output if any labels were missing.
        if missing_labels:
            raise RuntimeError(
                "Some retained images had no matching labels:\n" + "\n".join(missing_labels[:30])
            )

        # Verify image/label counts and no exact duplicate image contents remain.
        verification_hashes = set()
        verification_errors = []
        for split in SPLITS:
            image_dir = OUTPUT / split / "images"
            label_dir = OUTPUT / split / "labels"
            images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            labels = list(label_dir.glob("*.txt"))
            if len(images) != len(labels):
                verification_errors.append(
                    f"{split}: {len(images)} images but {len(labels)} label files"
                )
            for img in images:
                digest = sha256(img)
                if digest in verification_hashes:
                    verification_errors.append(f"Duplicate remained: {img}")
                verification_hashes.add(digest)
                if not (label_dir / f"{img.stem}.txt").is_file():
                    verification_errors.append(f"Missing label for: {img}")

        if verification_errors:
            raise RuntimeError("Verification failed:\n" + "\n".join(verification_errors))

        total_source = sum(len(items) for items in by_hash.values())
        total_kept = sum(kept.values())
        total_removed = sum(removed_by_split.values())

        lines = [
            "REMOVE ALL CONFLICTING DUPLICATE GROUPS REPORT",
            "=" * 55,
            f"Source dataset: {SOURCE}",
            f"Output dataset: {OUTPUT}",
            f"Unique image-content groups: {len(by_hash)}",
            f"Duplicate groups excluded entirely: {len(duplicate_groups)}",
            f"Images removed from duplicate groups: {duplicate_images}",
            f"Images copied to output: {total_kept}",
            f"Source images scanned: {total_source}",
            "",
            "Removed images by split:",
        ]
        lines += [f"  {split}: {removed_by_split[split]}" for split in SPLITS]
        lines += ["", "Retained images by split:"]
        lines += [f"  {split}: {kept[split]}" for split in SPLITS]
        lines += ["", "Excluded duplicate groups:"]
        for idx, (digest, items) in enumerate(sorted(duplicate_groups.items()), 1):
            lines.append(f"Group {idx} (SHA-256 {digest}):")
            lines += [f"  [{split}] {img}" for split, img in items]
        lines += ["", "Verification: PASSED", "No exact duplicate images remain across splits."]
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print("CLEANUP COMPLETED")
        print(f"Duplicate groups excluded entirely: {len(duplicate_groups)}")
        print(f"Images removed: {total_removed}")
        print(f"Images retained: {total_kept}")
        for split in SPLITS:
            print(f"{split.upper()}: kept {kept[split]}, removed {removed_by_split[split]}")
        print(f"\nNew dataset: {OUTPUT}")
        print(f"Report: {REPORT}")
        print("Verification: PASSED — no exact duplicate images remain.")
        print("Original deduplicated_dataset was not changed.")

    except Exception as e:
        # Remove incomplete output so it cannot be mistaken for a completed dataset.
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        sys.exit(f"ERROR: Cleanup failed; incomplete output was removed.\n{e}")

if __name__ == "__main__":
    main()
