# Pothole Detection Using YOLO26 and RF-DETR

A computer vision project focused on detecting potholes in road images using object detection models. The project includes dataset cleaning, annotation validation, duplicate detection, model training, and performance evaluation.

## Overview

Road potholes can affect road safety and vehicle conditions. This project explores object detection models to identify potholes in road images.

The project experiments with two approaches:
- **YOLO26n:** Lightweight object detection with checkpoint comparison and fine-tuning.
- **RF-DETR Nano:** Transformer-based object detection with multiple training experiments.

## Features

- Dataset cleaning and duplicate detection
- Annotation and label validation
- Background image preparation
- YOLO-format dataset preparation
- YOLO26 checkpoint comparison
- YOLO26 model fine-tuning
- RF-DETR training and evaluation
- Validation and test performance analysis

## Dataset

The project uses a cleaned pothole detection dataset in YOLO format.

Dataset splits:

| Split | Number of images |
|---|---:|
| Training | 22,791 |
| Validation | 3,256 |
| Testing | 2,245 |
| **Total** | **28,292** |

The dataset contains pothole images with object annotations. The dataset files are not included in this repository because of their size.

Expected directory structure:

```text
final_cleaned_dataset_no_conflicts/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml