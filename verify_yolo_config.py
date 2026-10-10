from pathlib import Path
import sys

ROOT = Path("/home/CL502-13/Downloads/latestmergedproject2.v1i.yolo26")
DATASET = ROOT / "final_cleaned_dataset_no_conflicts"
YAML_PATH = ROOT / "data_final_cleaned.yaml"
REPORT = ROOT / "training_config_check.txt"

EXPECTED = {
    "path": str(DATASET),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["pothole"],
}

def parse_simple_yaml(path):
    """Parse the exact small YAML format used by our generated data file."""
    parsed = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Cannot parse line: {raw_line}")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key in ("path", "train", "val", "test"):
            parsed[key] = value.strip("'\"")
        elif key == "nc":
            parsed[key] = int(value)
        elif key == "names":
            v = value.strip()
            if not (v.startswith("[") and v.endswith("]")):
                raise ValueError("Expected names in list form, e.g. names: ['pothole']")
            inner = v[1:-1].strip()
            parsed[key] = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
        else:
            parsed[key] = value
    return parsed

def main():
    errors = []
    notes = []
    if not YAML_PATH.is_file():
        sys.exit(f"ERROR: YAML file not found: {YAML_PATH}")
    try:
        config = parse_simple_yaml(YAML_PATH)
    except Exception as exc:
        sys.exit(f"ERROR: Could not read YAML: {exc}")

    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {config.get(key)!r}")

    resolved = {}
    for split in ("train", "val", "test"):
        rel = config.get(split)
        if not isinstance(rel, str):
            errors.append(f"{split}: path missing or invalid")
            continue
        folder = DATASET / rel
        resolved[split] = folder
        if not folder.is_dir():
            errors.append(f"{split}: image directory does not exist: {folder}")
        else:
            image_count = sum(
                1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
                }
            )
            notes.append(f"{split}: {folder} ({image_count} images)")

    # Verify corresponding label directories and label-file counts.
    split_to_name = {"train": "train", "val": "valid", "test": "test"}
    for split, split_name in split_to_name.items():
        image_dir = DATASET / config.get(split, "")
        label_dir = DATASET / split_name / "labels"
        if image_dir.is_dir() and label_dir.is_dir():
            images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
            }]
            labels = list(label_dir.glob("*.txt"))
            if len(images) != len(labels):
                errors.append(f"{split}: {len(images)} images but {len(labels)} label files")
            image_stems = {p.stem for p in images}
            label_stems = {p.stem for p in labels}
            if image_stems != label_stems:
                errors.append(f"{split}: image and label filenames do not match exactly")

    lines = [
        "YOLO DATA CONFIGURATION CHECK",
        "=" * 55,
        f"YAML: {YAML_PATH}",
        f"Dataset root: {DATASET}",
        f"Class count: {config.get('nc')}",
        f"Class names: {config.get('names')}",
        "",
        "Resolved splits:",
        *(notes if notes else ["No split paths resolved."]),
        "",
        "Errors:",
        *(errors if errors else ["None"]),
        "",
        "Status: " + ("FAILED — fix errors before training" if errors else "PASSED — configuration and paths match the cleaned dataset"),
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("YOLO CONFIGURATION CHECK FINISHED")
    print(f"YAML: {YAML_PATH}")
    print(f"Dataset root: {DATASET}")
    print(f"Class count: {config.get('nc')}")
    print(f"Class names: {config.get('names')}")
    for note in notes:
        print(note)
    print(f"Errors: {len(errors)}")
    print(f"Report: {REPORT}")
    if errors:
        print("STATUS: FAILED — do not train yet.")
        sys.exit(1)
    print("STATUS: PASSED — YAML and dataset paths are correct.")
    print("\nTRAINING CONFIGURATION TO USE (not started yet):")
    print(f"data={YAML_PATH}")
    print("model=yolo26n.pt")
    print("imgsz=640")
    print("batch=8 (reduce to 4 if GPU memory runs out)")
    print("epochs=100 (training duration; early stopping may finish sooner)")
    print("patience=20")
    print("device=0 (NVIDIA GPU; omit or change if unavailable)")
    print("\nNo training was started by this script.")

if __name__ == "__main__":
    main()
