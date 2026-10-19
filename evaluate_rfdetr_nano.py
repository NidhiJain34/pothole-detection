"""
RF-DETR Nano - Test Evaluation

Purpose:
    Evaluate the best RF-DETR Nano checkpoint from the
    512-resolution training experiment on the test dataset.

Dataset:
    final_cleaned_dataset_no_conflicts/

Model:
    runs/rfdetr/nano_512_resolution/checkpoint_best_ema.pth

Output:
    runs/rfdetr/nano_512_resolution/test_evaluation/metrics.json
"""

from pathlib import Path
import json
import sys

from rfdetr import RFDETRNano


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "final_cleaned_dataset_no_conflicts"

CHECKPOINT = (
    PROJECT_DIR
    / "runs"
    / "rfdetr"
    / "nano_512_resolution"
    / "checkpoint_best_ema.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "runs"
    / "rfdetr"
    / "nano_512_resolution"
    / "test_evaluation"
)

METRICS_FILE = OUTPUT_DIR / "metrics.json"


# ============================================================
# 2. VALIDATE FILES AND DIRECTORIES
# ============================================================

def validate_paths():
    """Check that the trained model and test dataset exist."""

    print("\nChecking project files...", flush=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"\nTrained checkpoint not found:\n{CHECKPOINT}\n"
            "Check that the 512-resolution training experiment "
            "has completed successfully."
        )

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"\nDataset directory not found:\n{DATASET_DIR}"
        )

    test_images = DATASET_DIR / "test" / "images"
    test_labels = DATASET_DIR / "test" / "labels"

    if not test_images.is_dir():
        raise FileNotFoundError(
            f"\nTest images directory not found:\n{test_images}"
        )

    if not test_labels.is_dir():
        raise FileNotFoundError(
            f"\nTest labels directory not found:\n{test_labels}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checkpoint found.", flush=True)
    print("Dataset found.", flush=True)
    print("Test images and labels found.", flush=True)


# ============================================================
# 3. LOAD THE TRAINED MODEL
# ============================================================

def load_model():
    """Load the best EMA checkpoint from the 512-resolution run."""

    print("\n" + "=" * 65, flush=True)
    print("LOADING TRAINED RF-DETR NANO MODEL", flush=True)
    print("=" * 65, flush=True)

    print(f"\nCheckpoint:\n{CHECKPOINT}", flush=True)

    # Load the trained model directly from the saved checkpoint.
    model = RFDETRNano.from_checkpoint(str(CHECKPOINT))

    print("\nModel loaded successfully.", flush=True)

    return model


# ============================================================
# 4. EVALUATE ON THE TEST DATASET
# ============================================================

def evaluate_model(model):
    """Evaluate the trained model on the existing test split."""

    print("\n" + "=" * 65, flush=True)
    print("RF-DETR NANO - 512 RESOLUTION TEST EVALUATION", flush=True)
    print("=" * 65, flush=True)

    print(f"\nDataset directory:\n{DATASET_DIR}", flush=True)
    print("Evaluation split: test", flush=True)

    print(
        "\nStarting evaluation. This may take some time...",
        flush=True,
    )

    # Evaluate using the existing YOLO-format dataset.
    # The test split is used only for final evaluation.
    metrics = model.evaluate(
        dataset_dir=str(DATASET_DIR),
        split="test",
        dataset_file="yolo",
    )

    return metrics


# ============================================================
# 5. SAVE AND DISPLAY EVALUATION METRICS
# ============================================================

def save_metrics(metrics):
    """Save the evaluation results to a JSON file."""

    print("\n" + "=" * 65, flush=True)
    print("TEST EVALUATION RESULTS", flush=True)
    print("=" * 65, flush=True)

    if metrics is None:
        print(
            "\nThe evaluation function did not return a metrics object.",
            flush=True,
        )
        print(
            "Check the evaluation output above for the reported metrics.",
            flush=True,
        )
        return

    if isinstance(metrics, dict):
        # Convert any non-standard values to strings if necessary.
        with METRICS_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                metrics,
                file,
                indent=4,
                default=str,
            )

        print("\nEvaluation metrics:", flush=True)
        print(
            json.dumps(metrics, indent=4, default=str),
            flush=True,
        )

        print(
            f"\nMetrics saved successfully to:\n{METRICS_FILE}",
            flush=True,
        )

    else:
        print("\nEvaluation output:", flush=True)
        print(metrics, flush=True)

        print(
            "\nThe returned result was not a dictionary, "
            "so it was not saved automatically as JSON.",
            flush=True,
        )


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():
    """Run the complete test evaluation pipeline."""

    print("\n" + "#" * 65, flush=True)
    print("RF-DETR NANO: FINAL TEST EVALUATION", flush=True)
    print("#" * 65, flush=True)

    try:
        # Step 1: Validate model and dataset paths.
        validate_paths()

        # Step 2: Load the trained checkpoint.
        model = load_model()

        # Step 3: Evaluate on the test split.
        metrics = evaluate_model(model)

        # Step 4: Save and display the results.
        save_metrics(metrics)

        print("\n" + "#" * 65, flush=True)
        print("TEST EVALUATION COMPLETED", flush=True)
        print("#" * 65, flush=True)

    except KeyboardInterrupt:
        print(
            "\nEvaluation interrupted by the user.",
            flush=True,
        )
        sys.exit(1)

    except Exception as error:
        print(
            "\nERROR DURING TEST EVALUATION",
            flush=True,
        )
        print(f"{type(error).__name__}: {error}", flush=True)

        raise


if __name__ == "__main__":
    main()