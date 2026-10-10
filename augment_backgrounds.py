from pathlib import Path
import random
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

SOURCE_IMAGES = ROOT / "train" / "images"
SOURCE_LABELS = ROOT / "train" / "labels"

OUTPUT_ROOT = ROOT / "generated_background"
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"

TARGET_COUNT = 4191
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
rng = np.random.default_rng(SEED)


def find_backgrounds():
    """Find readable training images whose label files are empty."""
    backgrounds = []

    for image_path in SOURCE_IMAGES.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise RuntimeError(
                f"Missing label file for {image_path.name}"
            )

        if label_path.read_text(encoding="utf-8").strip():
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            raise RuntimeError(
                f"Unreadable source image: {image_path.name}"
            ) from exc

        backgrounds.append(image_path)

    return sorted(backgrounds)


def augment_image(image_path, variant_number):
    """Create a moderate visual variation of a background image."""
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    # Mild brightness and contrast changes.
    image = ImageEnhance.Brightness(image).enhance(
        random.uniform(0.88, 1.12)
    )
    image = ImageEnhance.Contrast(image).enhance(
        random.uniform(0.90, 1.10)
    )

    # Horizontal flip is valid for ordinary road scenes.
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Small rotation, filling exposed edges using nearby pixels.
    angle = random.uniform(-4.0, 4.0)
    image = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=None
    )

    # Occasionally apply a very mild blur.
    if random.random() < 0.08:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=0.35)
        )

    # Occasionally add very mild sensor-like noise.
    if random.random() < 0.12:
        array = np.asarray(image).astype(np.float32)
        noise = rng.normal(0, 1.5, array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)

    return image


def main():
    print("=" * 60)
    print("TRAINING BACKGROUND AUGMENTATION")
    print("=" * 60)

    if not SOURCE_IMAGES.is_dir() or not SOURCE_LABELS.is_dir():
        raise RuntimeError("Training images or labels folder is missing.")

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(
            f"Output folder is not empty:\n{OUTPUT_ROOT}\n"
            "Rename or inspect it before running this script again."
        )

    backgrounds = find_backgrounds()

    print(f"Background source images: {len(backgrounds)}")

    if not backgrounds:
        raise RuntimeError("No background images were found.")

    if TARGET_COUNT < 1:
        raise ValueError("TARGET_COUNT must be positive.")

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    created = 0

    try:
        for index in range(TARGET_COUNT):
            source = backgrounds[index % len(backgrounds)]
            variant = index + 1

            # Unique filenames, separate from the originals.
            filename = f"aug_bg_{variant:05d}.jpg"
            image_path = OUTPUT_IMAGES / filename
            label_path = OUTPUT_LABELS / f"aug_bg_{variant:05d}.txt"

            image = augment_image(source, variant)
            image.save(
                image_path,
                format="JPEG",
                quality=95,
                optimize=True
            )

            # Background-only image: no YOLO object annotations.
            label_path.write_text("", encoding="utf-8")

            created += 1

            if created % 250 == 0 or created == TARGET_COUNT:
                print(f"Generated {created}/{TARGET_COUNT}")

        # Verify output counts.
        generated_images = [
            p for p in OUTPUT_IMAGES.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        generated_labels = list(OUTPUT_LABELS.glob("*.txt"))

        empty_labels = sum(
            1 for path in generated_labels
            if not path.read_text(encoding="utf-8").strip()
        )

        print("\n" + "=" * 60)
        print("GENERATION SUMMARY")
        print("=" * 60)
        print(f"Images generated: {len(generated_images)}")
        print(f"Label files created: {len(generated_labels)}")
        print(f"Empty labels verified: {empty_labels}")

        if (
            len(generated_images) != TARGET_COUNT
            or len(generated_labels) != TARGET_COUNT
            or empty_labels != TARGET_COUNT
        ):
            raise RuntimeError(
                "Output verification failed. Do not merge these files."
            )

        print("\nSUCCESS: All generated files passed count verification.")
        print("Original dataset and validation/test splits were not changed.")
        print(f"Output folder: {OUTPUT_ROOT}")

    except Exception:
        print(
            "\nGeneration stopped because an error occurred. "
            "Inspect the staging folder before trying again."
        )
        raise


if __name__ == "__main__":
    main()