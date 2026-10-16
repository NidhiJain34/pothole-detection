from pathlib import Path
import torch
from rfdetr import RFDETRNano


# ==========================================
# RF-DETR NANO - POTHOLE DETECTION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "final_cleaned_dataset_no_conflicts"

OUTPUT_DIR = PROJECT_DIR / "runs" / "rfdetr" / "nano_10_epochs"


def main():

    print("=" * 60)
    print("RF-DETR NANO - INDEPENDENT TRAINING")
    print("=" * 60)

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Please check your GPU setup."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Dataset:", DATASET_DIR)
    print("Output:", OUTPUT_DIR)

    # Check dataset configuration
    if not (DATASET_DIR / "data.yaml").is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: "
            f"{DATASET_DIR / 'data.yaml'}"
        )

    # Check dataset folders
    for split in ["train", "valid"]:
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing images or labels for {split}"
            )

    # Create a separate output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load RF-DETR Nano with its own pretrained weights
    print("\nLoading RF-DETR Nano...")
    model = RFDETRNano()

    # Begin training
    print("\nStarting training...")
    print("Model: RF-DETR Nano")
    print("Epochs: 10")
    print("Batch size: 4")
    print("Gradient accumulation: 4")
    print("Learning rate: 0.0001")
    print("Dataset: Existing cleaned pothole dataset")
    print("YOLO weights: Not used")
    print("-" * 60)

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=str(OUTPUT_DIR),
    )

    print("\nTraining completed!")
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()