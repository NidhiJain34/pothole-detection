from pathlib import Path
import json
from rfdetr import RFDETRNano


# ==========================================
# RF-DETR NANO - TEST EVALUATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "final_cleaned_dataset_no_conflicts"

CHECKPOINT = (
    PROJECT_DIR
    / "runs"
    / "rfdetr"
    / "nano_10_epochs"
    / "checkpoint_best_ema.pth"
)

OUTPUT_DIR = PROJECT_DIR / "runs" / "rfdetr" / "test_evaluation"


def main():
    print("=" * 60)
    print("RF-DETR NANO - TEST SET EVALUATION")
    print("=" * 60)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {CHECKPOINT}"
        )

    if not (DATASET_DIR / "test" / "images").is_dir():
        raise FileNotFoundError("Test images directory not found.")

    if not (DATASET_DIR / "test" / "labels").is_dir():
        raise FileNotFoundError("Test labels directory not found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checkpoint:", CHECKPOINT)
    print("Test dataset:", DATASET_DIR / "test")
    print("Output directory:", OUTPUT_DIR)

    # Load the trained RF-DETR Nano weights.
    print("\nLoading trained RF-DETR Nano...")
    model = RFDETRNano(
        pretrain_weights=str(CHECKPOINT)
    )

    # Evaluate against the existing test split.
    print("\nEvaluating on the test split...")
    metrics = model.evaluate(
        dataset_dir=str(DATASET_DIR),
        split="test",
        dataset_file="yolo",
    )

    # Save evaluation results.
    if isinstance(metrics, dict):
        results_file = OUTPUT_DIR / "metrics.json"

        with results_file.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4, default=str)

        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=4, default=str))
        print("\nMetrics saved to:", results_file)
    else:
        print("\nEvaluation returned:", metrics)

    print("\nTest evaluation finished.")


if __name__ == "__main__":
    main()