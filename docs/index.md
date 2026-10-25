
# Automated Pork Packaging Inspection

## Project Overview

This project investigates automated detection of pork rasher
packaging defects and visual meat presentation errors using
computer vision.

The project compares two computational approaches:

1. Histogram of Oriented Gradients (HOG) with Support Vector
   Machine (SVM) classification.
2. YOLO11n deep-learning object detection.

## Inspection Categories

The system evaluates five defect categories:

- Loose meat
- Packaging error
- Twisted meat
- Unsealed packaging
- Wrinkle

These categories relate to packaging quality and visible meat
presentation. The system does not evaluate freshness or food safety.

## Main Findings

The extended YOLO11n model achieved the following performance
on the held-out test dataset:

| Metric | Result |
|---|---:|
| Precision | 47.44% |
| Recall | 51.64% |
| mAP@0.5 | 55.80% |
| mAP@0.5:0.95 | 25.50% |

Although YOLO outperformed the HOG + SVM baseline during
validation, missed defects and reduced test performance indicate
that further improvements are required before industrial deployment.

## Project Documentation

Use the navigation menu to explore:

- Dataset preparation and validation
- Computational methods and assumptions
- Model development and optimisation
- Evaluation results and limitations
- Reproducibility instructions
