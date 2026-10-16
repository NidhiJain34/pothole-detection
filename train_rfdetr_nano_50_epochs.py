from pathlib import Path
from rfdetr import RFDETRNano

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "final_cleaned_dataset_no_conflicts"
CHECKPOINT = PROJECT_DIR / "runs/rfdetr/nano_10_epochs/checkpoint_best_ema.pth"
OUTPUT_DIR = PROJECT_DIR / "runs/rfdetr/nano_50_epochs"

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("RF-DETR NANO: 50-EPOCH FINE-TUNING", flush=True)
print("=" * 60, flush=True)
print(f"Dataset: {DATASET_DIR}", flush=True)
print(f"Starting checkpoint: {CHECKPOINT}", flush=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

model = RFDETRNano.from_checkpoint(str(CHECKPOINT))

model.train(
    dataset_dir=str(DATASET_DIR),
    dataset_file="yolo",
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-5,
    output_dir=str(OUTPUT_DIR),
)

print("Training completed!", flush=True)
print(f"Results saved in: {OUTPUT_DIR}", flush=True)
