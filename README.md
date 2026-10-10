
# Pothole Detection Using YOLO26

## Overview
A computer vision project that detects potholes in road images using the YOLO26 object detection model.

## Features
- Dataset cleaning and duplicate detection
- Annotation and label validation
- Background image preparation
- YOLO dataset configuration
- Model training and evaluation

## Dataset
The dataset contains pothole images and background images without potholes. The dataset is excluded from this repository because of its large size.

## Model
YOLO26n trained on a cleaned pothole dataset.

## Test Results
The model was evaluated on 2,245 test images.

| Metric | Result |
|---|---:|
| Precision | 76.4% |
| Recall | 59.7% |
| mAP@50 | 67.4% |
| mAP@50–95 | 33.9% |

## Requirements
- Python 3.11
- PyTorch
- Ultralytics
- CUDA-compatible GPU recommended

## Usage
1. Clone this repository.
2. Install the required dependencies.
3. Prepare the dataset in YOLO format.
4. Update the dataset paths if necessary.
5. Run the training or evaluation scripts.

## Notes
The dataset, trained model weights, virtual environment, and training outputs are excluded from GitHub.
# pothole-detection
A pothole detection project focused on road damage identification using computer vision, dataset preprocessing, image and annotation validation, duplicate detection, and machine learning model training and evaluation.
