from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "deduplicated_dataset"
OUTPUT = ROOT / "annotation_previews"

SPLIT = "train"
NUM_IMAGES = 20
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    image_dir = DATASET / SPLIT / "images"
    label_dir = DATASET / SPLIT / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print("ERROR: Dataset images or labels folder not found.")
        print(f"Expected images: {image_dir}")
        print(f"Expected labels: {label_dir}")
        return

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("ERROR: No images found.")
        return

    random.seed(RANDOM_SEED)
    selected = random.sample(images, min(NUM_IMAGES, len(images)))

    OUTPUT.mkdir(parents=True, exist_ok=True)

    successful = 0
    missing_labels = 0

    for index, image_path in enumerate(selected, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Missing label: {image_path.name}")
            missing_labels += 1
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size

            # Reopen the image so the drawing uses an independent copy.
            with Image.open(image_path) as source:
                preview = source.convert("RGB")

            draw = ImageDraw.Draw(preview)

            boxes_drawn = 0

            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            for line in lines:
                values = line.split()

                if not values:
                    continue

                if len(values) != 5:
                    print(
                        f"Skipping malformed line in {label_path.name}: "
                        f"{line}"
                    )
                    continue

                try:
                    class_id = int(values[0])
                    x_center, y_center, box_width, box_height = map(
                        float, values[1:]
                    )
                except ValueError:
                    print(f"Invalid numbers in {label_path.name}")
                    continue

                # This project is pothole-only, so class 0 is pothole.
                if class_id != 0:
                    continue

                # YOLO coordinates are normalized from 0 to 1.
                x1 = int((x_center - box_width / 2) * width)
                y1 = int((y_center - box_height / 2) * height)
                x2 = int((x_center + box_width / 2) * width)
                y2 = int((y_center + box_height / 2) * height)

                # Clip coordinates to image boundaries.
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline="red",
                    width=max(2, width // 400),
                )

                draw.text(
                    (x1, max(0, y1 - 18)),
                    "pothole",
                    fill="red",
                )

                boxes_drawn += 1

            output_path = OUTPUT / f"preview_{index:02d}_{image_path.name}"
            preview.save(output_path)

            successful += 1

            print(
                f"{successful}. {image_path.name} "
                f"| Boxes drawn: {boxes_drawn} "
                f"| Preview: {output_path.name}"
            )

        except Exception as error:
            print(f"Could not process {image_path.name}: {error}")

    print("\n========== PREVIEW RESULTS ==========")
    print(f"Images processed: {successful}")
    print(f"Missing labels: {missing_labels}")
    print(f"Preview folder: {OUTPUT}")
    print("\nNo original images or labels were modified.")


if __name__ == "__main__":
    main()