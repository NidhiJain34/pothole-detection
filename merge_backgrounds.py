from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

GENERATED_IMAGES = ROOT / "generated_background" / "images"
GENERATED_LABELS = ROOT / "generated_background" / "labels"

TRAIN_IMAGES = ROOT / "train" / "images"
TRAIN_LABELS = ROOT / "train" / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_COUNT = 4191


def main():
    print("=" * 60)
    print("MERGING AUGMENTED BACKGROUND IMAGES")
    print("=" * 60)

    # Check all required folders exist.
    required_folders = [
        GENERATED_IMAGES,
        GENERATED_LABELS,
        TRAIN_IMAGES,
        TRAIN_LABELS,
    ]

    for folder in required_folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

    # Collect generated images.
    images = sorted(
        p for p in GENERATED_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} generated images, "
            f"but found {len(images)}. Nothing was merged."
        )

    # Verify matching, empty labels.
    for image in images:
        label = GENERATED_LABELS / f"{image.stem}.txt"

        if not label.is_file():
            raise RuntimeError(
                f"Missing label for {image.name}. Nothing was merged."
            )

        if label.read_text(encoding="utf-8").strip():
            raise RuntimeError(
                f"Expected an empty background label: {label.name}"
            )

    # Check for filename collisions before copying anything.
    conflicts = []

    for image in images:
        if (TRAIN_IMAGES / image.name).exists():
            conflicts.append(str(TRAIN_IMAGES / image.name))

        if (TRAIN_LABELS / f"{image.stem}.txt").exists():
            conflicts.append(str(TRAIN_LABELS / f"{image.stem}.txt"))

    if conflicts:
        print("Filename conflicts found. Nothing was merged.")
        for conflict in conflicts[:20]:
            print(" -", conflict)
        raise RuntimeError("Resolve filename conflicts before proceeding.")

    print(f"Generated images verified: {len(images)}")
    print("All corresponding labels exist and are empty.")
    print("No filename conflicts found.")
    print("\nStarting merge...")

    copied_images = 0
    copied_labels = 0

    try:
        for index, image in enumerate(images, start=1):
            destination_image = TRAIN_IMAGES / image.name
            destination_label = TRAIN_LABELS / f"{image.stem}.txt"

            shutil.copy2(image, destination_image)
            copied_images += 1

            shutil.copy2(
                GENERATED_LABELS / f"{image.stem}.txt",
                destination_label,
            )
            copied_labels += 1

            if index % 250 == 0 or index == EXPECTED_COUNT:
                print(f"Merged {index}/{EXPECTED_COUNT}")

    except Exception:
        print("\nERROR during merge.")
        print(f"Images copied: {copied_images}")
        print(f"Labels copied: {copied_labels}")
        print("Inspect the training folders before rerunning.")
        raise

    print("\n" + "=" * 60)
    print("MERGE COMPLETED")
    print("=" * 60)
    print(f"Images copied: {copied_images}")
    print(f"Empty labels copied: {copied_labels}")
    print("Validation and test folders were not modified.")
    print("\nNext step: verify the final training counts.")


if __name__ == "__main__":
    main()