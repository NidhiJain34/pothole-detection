from pathlib import Path
from rfdetr import RFDETRNano

# ============================================================
# RF-DETR NANO: HIGHER-RESOLUTION FINE-TUNING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "final_cleaned_dataset_no_conflicts"

CHECKPOINT = (
    PROJECT_DIR
    / "runs"
    / "rfdetr"
    / "nano_50_epochs"
    / "checkpoint_best_ema.pth"
)

OUTPUT_DIR = PROJECT_DIR / "runs" / "rfdetr" / "nano_512_resolution"

# Training configuration
RESOLUTION = 512
MAX_EPOCHS = 100
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 5e-6

# Early stopping configuration
PATIENCE = 10
MIN_DELTA = 0.001

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

if RESOLUTION % (16 * 2) != 0:
    raise ValueError(
        "Resolution must be divisible by patch_size * num_windows."
    )

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_DIR}\n"
        "Choose a new output directory to preserve existing results."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65, flush=True)
print("RF-DETR NANO - 512 RESOLUTION EXPERIMENT", flush=True)
print("=" * 65, flush=True)

print(f"Dataset: {DATASET_DIR}", flush=True)
print(f"Starting checkpoint: {CHECKPOINT}", flush=True)
print(f"Resolution: {RESOLUTION} x {RESOLUTION}", flush=True)
print(f"Maximum epochs: {MAX_EPOCHS}", flush=True)
print(f"Batch size: {BATCH_SIZE}", flush=True)
print(f"Gradient accumulation: {GRAD_ACCUM_STEPS}", flush=True)
print(f"Learning rate: {LEARNING_RATE}", flush=True)
print(f"Early stopping patience: {PATIENCE}", flush=True)
print(f"Early stopping min delta: {MIN_DELTA}", flush=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# ============================================================
# LOAD CHECKPOINT
# ============================================================

model = RFDETRNano.from_checkpoint(str(CHECKPOINT))

print("\nCheckpoint loaded.", flush=True)
print(
    f"Loaded model resolution: {model.model_config.resolution}",
    flush=True,
)
print(
    f"Loaded positional encoding size: "
    f"{model.model_config.positional_encoding_size}",
    flush=True,
)

# ============================================================
# TRAIN
# ============================================================

model.train(
    dataset_dir=str(DATASET_DIR),
    dataset_file="yolo",

    # Higher input resolution
    resolution=RESOLUTION,

    # Training duration
    epochs=MAX_EPOCHS,

    # Memory-conscious batch configuration
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,

    # Fine-tuning
    lr=LEARNING_RATE,

    # Select and stop based on mAP@50:95
    best_model_metric="map",
    early_stopping=True,
    early_stopping_patience=PATIENCE,
    early_stopping_min_delta=MIN_DELTA,
    early_stopping_use_ema=True,

    # Keep results separate
    output_dir=str(OUTPUT_DIR),
)

print("\nTraining finished.", flush=True)
print(f"Results saved to: {OUTPUT_DIR}", flush=True)