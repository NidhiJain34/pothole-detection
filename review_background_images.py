#!/usr/bin/env python3
"""Create read-only contact sheets of empty-label/background images for review."""

from pathlib import Path
import random
import math
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit('Pillow missing. Install with the project venv: .venv/bin/python -m pip install Pillow')

ROOT = Path("/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26")
DATASET = ROOT / "final_cleaned_dataset_no_conflicts"
OUT = ROOT / "background_review_contact_sheets"
PER_SHEET = 24
COLS = 4
THUMB_SIZE = (300, 210)
CELL_W, CELL_H = 330, 260
SEED = 2026
MAX_IMAGES = 120

def font(size=13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def main():
    if not DATASET.is_dir():
        sys.exit(f"Dataset folder not found: {DATASET}")

    backgrounds = []
    for split in ("train", "valid", "test"):
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"
        for img_path in image_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
            }:
                continue
            label_path = label_dir / img_path.relative_to(image_dir).with_suffix(".txt")
            if label_path.is_file():
                try:
                    if not label_path.read_text(encoding="utf-8").strip():
                        backgrounds.append((split, img_path, label_path))
                except OSError:
                    pass

    if not backgrounds:
        sys.exit("No empty-label images found.")

    rng = random.Random(SEED)
    rng.shuffle(backgrounds)
    sample = backgrounds[:min(MAX_IMAGES, len(backgrounds))]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Empty-label images found: {len(backgrounds):,}")
    print(f"Creating review sheets for {len(sample):,} sampled backgrounds.")
    made = []

    for sheet_idx in range(math.ceil(len(sample) / PER_SHEET)):
        batch = sample[sheet_idx * PER_SHEET:(sheet_idx + 1) * PER_SHEET]
        rows = math.ceil(len(batch) / COLS)
        sheet = Image.new("RGB", (COLS * CELL_W, rows * CELL_H), "white")
        draw = ImageDraw.Draw(sheet)

        for i, (split, img_path, label_path) in enumerate(batch):
            x = (i % COLS) * CELL_W
            y = (i // COLS) * CELL_H
            try:
                with Image.open(img_path) as src:
                    im = src.convert("RGB")
                    original_w, original_h = im.size
                    im.thumbnail(THUMB_SIZE)
                    px = x + (CELL_W - im.width) // 2
                    sheet.paste(im, (px, y + 5))
                draw.text((x + 6, y + 220), f"{split.upper()} | {original_w}x{original_h}", fill="black", font=font(14))
                draw.text((x + 6, y + 241), img_path.name[:43], fill="black", font=font(10))
            except Exception as exc:
                draw.text((x + 6, y + 20), f"Could not open: {img_path.name}", fill="black", font=font(11))
                draw.text((x + 6, y + 40), str(exc)[:45], fill="black", font=font(10))

        out_path = OUT / f"background_review_{sheet_idx + 1:02d}.jpg"
        sheet.save(out_path, quality=90)
        made.append(out_path)

    summary = OUT / "background_review_summary.txt"
    summary.write_text(
        "BACKGROUND IMAGE REVIEW CONTACT SHEETS\n"
        "======================================\n"
        f"Dataset: {DATASET}\n"
        f"Total empty-label images: {len(backgrounds):,}\n"
        f"Random sample shown: {len(sample):,}\n\n"
        "Review each image and check whether it truly contains no pothole.\n"
        "If a pothole is visible in an empty-label image, note its filename for review.\n"
        "This script does not modify any dataset files.\n\n"
        + "\n".join(str(p) for p in made) + "\n",
        encoding="utf-8"
    )
    print("\nCreated:")
    for p in made:
        print(p)
    print(summary)
    print("DONE — dataset files were not changed.")

if __name__ == "__main__":
    main()
