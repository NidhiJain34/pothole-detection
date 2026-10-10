from pathlib import Path

root = Path(__file__).parent

total_images = 0
total_background = 0
total_annotated = 0
total_missing = 0

for split in ["train", "valid", "test"]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print(f"{split}: folders not found")
        continue

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    ]

    background = 0
    annotated = 0
    missing = 0

    for image in images:
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing += 1
        elif not label.read_text().strip():
            background += 1
        else:
            annotated += 1

    total_images += len(images)
    total_background += background
    total_annotated += annotated
    total_missing += missing

    print(f"\n{split.upper()}")
    print(f"Total images: {len(images)}")
    print(f"Pothole-annotated images: {annotated}")
    print(f"Background images: {background}")
    print(f"Missing labels: {missing}")

print("\n===== DATASET TOTAL =====")
print(f"Total images: {total_images}")
print(f"Pothole-annotated images: {total_annotated}")
print(f"Background images: {total_background}")
print(f"Missing labels: {total_missing}")