from pathlib import Path
from rfdetr import RFDETRNano


# ============================================================
# PATHS
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

# NEW DIRECTORY: preserves all previous experiments
OUTPUT_DIR = (
    PROJECT_DIR
    / "runs"
    / "rfdetr"
    / "learning_rate_experiment_2"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RESOLUTION = 512
MAX_EPOCHS = 60

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8

# Lower learning rates for cautious fine-tuning
LEARNING_RATE = 5e-6
ENCODER_LEARNING_RATE = 5e-5

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.001


# ============================================================
# SAFETY CHECKS
# ============================================================

print("=" * 65)
print("RF-DETR NANO - LEARNING RATE EXPERIMENT 2")
print("=" * 65)

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATASET_DIR}"
    )

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Starting checkpoint not found:\n{CHECKPOINT}"
    )

for split in ("train", "valid", "test"):
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images directory:\n{images_dir}"
        )

    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Missing labels directory:\n{labels_dir}"
        )

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty:\n{OUTPUT_DIR}\n"
        "Previous results have been preserved. "
        "Choose a new output directory."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print(f"Dataset:              {DATASET_DIR}")
print(f"Starting checkpoint:  {CHECKPOINT}")
print(f"Output directory:     {OUTPUT_DIR}")
print(f"Resolution:           {RESOLUTION}")
print(f"Maximum epochs:       {MAX_EPOCHS}")
print(f"Batch size:           {BATCH_SIZE}")
print(f"Gradient accumulation:{GRAD_ACCUM_STEPS}")
print(f"Main learning rate:   {LEARNING_RATE}")
print(f"Encoder learning rate:{ENCODER_LEARNING_RATE}")
print(f"Early stopping:       {PATIENCE} epochs")
print("=" * 65)


# ============================================================
# INITIALIZE MODEL FROM CHECKPOINT
# ============================================================

print("\nLoading RF-DETR Nano from the selected checkpoint...")

model = RFDETRNano(
    pretrain_weights=str(CHECKPOINT),
    resolution=RESOLUTION,
    num_classes=1,
    trust_checkpoint=True,
)

print("Model initialized.")
print("Starting fine-tuning...\n")


# ============================================================
# TRAIN
# ============================================================

model.train(
    dataset_dir=str(DATASET_DIR),
    dataset_file="yolo",

    resolution=RESOLUTION,
    epochs=MAX_EPOCHS,

    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,

    lr=LEARNING_RATE,
    lr_encoder=ENCODER_LEARNING_RATE,

    best_model_metric="map",

    early_stopping=True,
    early_stopping_patience=PATIENCE,
    early_stopping_min_delta=MIN_DELTA,
    early_stopping_use_ema=True,

    output_dir=str(OUTPUT_DIR),
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINISHED")
print("=" * 65)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Best checkpoint:  {OUTPUT_DIR / 'checkpoint_best_ema.pth'}")
print("\nNext step: evaluate the best checkpoint on the test set.")