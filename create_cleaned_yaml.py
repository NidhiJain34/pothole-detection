from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "deduplicated_dataset"
OUTPUT = ROOT / "data_cleaned.yaml"


def main():
    required_paths = [
        DATASET / "train" / "images",
        DATASET / "train" / "labels",
        DATASET / "valid" / "images",
        DATASET / "valid" / "labels",
        DATASET / "test" / "images",
        DATASET / "test" / "labels",
    ]

    # Verify the dataset folders exist.
    for path in required_paths:
        if not path.is_dir():
            print(f"ERROR: Missing directory:\n{path}")
            return

    # Create YOLO configuration without requiring PyYAML.
    config = f"""path: '{DATASET.resolve()}'
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['pothole']
"""

    OUTPUT.write_text(config, encoding="utf-8")

    print("Cleaned YOLO configuration created successfully!")
    print(f"\nFile: {OUTPUT}")
    print("\nConfiguration:")
    print(config)


if __name__ == "__main__":
    main()