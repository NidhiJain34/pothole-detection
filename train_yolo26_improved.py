from ultralytics import YOLO
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Cleaned dataset configuration
DATASET = PROJECT_DIR / "final_cleaned_dataset_no_conflicts" / "data.yaml"

# Start from our best existing checkpoint
MODEL = (
    PROJECT_DIR
    / "runs/detect/runs/pothole/continued_training_90/weights/best.pt"
)

# Output directory for this experiment
PROJECT_NAME = "yolo26_improved_800"

def main():
    # Check that the required files exist
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET}")

    if not MODEL.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL}")

    print("=" * 60)
    print("YOLO26 IMPROVED FINE-TUNING")
    print("=" * 60)
    print(f"Starting checkpoint: {MODEL}")
    print(f"Dataset: {DATASET}")
    print("Image size: 800")
    print("Maximum epochs: 60")
    print("Batch size: 8")
    print("=" * 60)

    # Load the existing trained model
    model = YOLO(str(MODEL))

    # Fine-tune the model
    results = model.train(
        data=str(DATASET),

        # Training duration
        epochs=60,
        patience=12,

        # Image and batch settings
        imgsz=800,
        batch=8,

        # Optimizer and learning rate
        optimizer="MuSGD",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Data loading
        workers=4,
        cache=False,

        # GPU and numerical precision
        device=0,
        amp=True,

        # Data augmentation
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.5,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Save the new experiment separately
        project=str(PROJECT_DIR / "runs/detect"),
        name=PROJECT_NAME,
        exist_ok=False,
        save=True,
        plots=True,
        val=True,
    )

    print("\nTraining finished!")
    print(f"Results directory: {results.save_dir}")
    print(f"Best checkpoint: {results.save_dir}/weights/best.pt")
    print(f"Last checkpoint: {results.save_dir}/weights/last.pt")


if __name__ == "__main__":
    main()