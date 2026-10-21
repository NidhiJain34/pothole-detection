from pathlib import Path
import csv
import torch
from ultralytics import YOLO

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Validation dataset
DATASET = PROJECT_DIR / "final_cleaned_dataset_no_conflicts" / "data.yaml"

# All existing YOLO checkpoints
CHECKPOINTS = [
    PROJECT_DIR / "runs/detect/runs/pothole/smoke_test/weights/best.pt",
    PROJECT_DIR / "runs/detect/runs/pothole/train_10_epochs_batch16/weights/best.pt",
    PROJECT_DIR / "runs/detect/runs/pothole/continued_training_90/weights/best.pt",
    PROJECT_DIR / "runs/detect/runs/pothole/fine_tune_lr001_50_sgd/weights/best.pt",
    PROJECT_DIR / "runs/detect/train/weights/best.pt",
]

# Save comparison results separately
OUTPUT_DIR = PROJECT_DIR / "runs/detect/yolo_checkpoint_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATASET}")

results = []

for checkpoint in CHECKPOINTS:
    if not checkpoint.exists():
        print(f"\nSkipping missing checkpoint: {checkpoint}")
        continue

    print("\n" + "=" * 70)
    print(f"Evaluating: {checkpoint}")
    print("=" * 70)

    try:
        model = YOLO(str(checkpoint))

        # Evaluate on validation data, not test data
        metrics = model.val(
            data=str(DATASET),
            split="val",
            imgsz=640,
            batch=8,
            device=0,
            plots=False,
            verbose=False,
        )

        row = {
            "checkpoint": str(checkpoint.relative_to(PROJECT_DIR)),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

        results.append(row)

        print(f"mAP@50:     {row['mAP50']:.4f}")
        print(f"mAP@50:95:  {row['mAP50_95']:.4f}")
        print(f"Precision:  {row['precision']:.4f}")
        print(f"Recall:     {row['recall']:.4f}")

    except Exception as error:
        print(f"Evaluation failed for {checkpoint}")
        print(f"Error: {error}")

# Save results
csv_path = OUTPUT_DIR / "comparison.csv"

if results:
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("CHECKPOINT COMPARISON")
    print("=" * 70)

    for row in sorted(
        results,
        key=lambda item: item["mAP50_95"],
        reverse=True,
    ):
        print(
            f"\n{row['checkpoint']}"
            f"\n  mAP@50:95: {row['mAP50_95']:.4f}"
            f"\n  mAP@50:    {row['mAP50']:.4f}"
            f"\n  Precision: {row['precision']:.4f}"
            f"\n  Recall:    {row['recall']:.4f}"
        )

    print(f"\nResults saved to: {csv_path}")
else:
    print("\nNo checkpoints were evaluated successfully.")