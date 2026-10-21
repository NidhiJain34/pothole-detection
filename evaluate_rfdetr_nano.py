from pathlib import Path
import json
import torch
from rfdetr import RFDETRNano

# Get the project folder
PROJECT_DIR = Path(__file__).resolve().parent

# Paths
DATASET_DIR = PROJECT_DIR / "final_cleaned_dataset_no_conflicts"
CHECKPOINT = (
    PROJECT_DIR
    / "runs"
    / "rfdetr"
    / "learning_rate_experiment_2"
    / "checkpoint_best_ema.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "runs"
    / "rfdetr"
    / "learning_rate_experiment_2"
    / "test_evaluation"
)

# Check that the model exists
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT}"
    )

# Check that the test images and labels exist
TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_LABELS = DATASET_DIR / "test" / "labels"

if not TEST_IMAGES.exists():
    raise FileNotFoundError(
        f"Test images folder not found: {TEST_IMAGES}"
    )

if not TEST_LABELS.exists():
    raise FileNotFoundError(
        f"Test labels folder not found: {TEST_LABELS}"
    )

# Create the output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RF-DETR TEST EVALUATION")
print("=" * 60)

print(f"\nCheckpoint: {CHECKPOINT}")
print(f"Dataset:    {DATASET_DIR}")
print(f"Output:     {OUTPUT_DIR}")

print("\nLoading trained RF-DETR Nano model...")

# Load the trained checkpoint
model = RFDETRNano.from_checkpoint(str(CHECKPOINT))

print("Model loaded successfully.")

print("\nStarting evaluation on the test dataset...")
print("This evaluates the model; it does not train it again.")

# Evaluate on the test split
results = model.evaluate(
    dataset_dir=str(DATASET_DIR),
    split="test",
    batch_size=2,
)

print("\nEvaluation finished.")
print("\nResults returned by RF-DETR:")
print(results)

# Save metrics if returned as a dictionary
if isinstance(results, dict):
    metrics = results
elif hasattr(results, "model_dump"):
    metrics = results.model_dump()
elif hasattr(results, "__dict__"):
    metrics = results.__dict__
else:
    metrics = {"results": str(results)}

metrics_path = OUTPUT_DIR / "metrics.json"

# Convert any unusual values into JSON-compatible values
def make_json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass

    if isinstance(value, Path):
        return str(value)

    return value


with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        make_json_safe(metrics),
        file,
        indent=4,
        default=str,
    )

print("\n" + "=" * 60)
print("TEST EVALUATION COMPLETED")
print("=" * 60)
print(f"Metrics saved to:\n{metrics_path}")