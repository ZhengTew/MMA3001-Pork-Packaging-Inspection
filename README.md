# Automated Pork Packaging Defect Detection

## Documentation
Detailed technical documentation is provided using MkDocs.

The documentation includes:

- Dataset exploration, cleaning and preparation.
- HOG + SVM computational methodology.
- YOLO11n training and optimisation.
- Model evaluation and experimental results.
- Reproducibility instructions and limitations.

### View Documentation Locally
Install dependencies:

    python -m pip install -r requirements.txt

Start the documentation server:

    python -m mkdocs serve

Open http://127.0.0.1:8000/ in your browser.

### Online Documentation

The HTML documentation is published through GitHub Pages.

[View Technical Documentation](http://127.0.0.1:8000/)

## 1. Project Overview

This project develops and evaluates a computer vision system for detecting packaging defects and visual presentation errors in pork rasher products.

The system uses RGB images and identifies five defect categories:

- Loose meat
- Packaging error
- Twisted meat
- Unsealed packaging
- Wrinkle

The project compares a traditional Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM) baseline against a YOLO11n deep-learning object detector.

**Scope:** The system evaluates visible packaging and product presentation defects. It does not assess meat freshness, microbiological safety or spoilage.

## 2. Project Objectives

1. Explore and prepare an annotated pork packaging image dataset.
2. Implement a HOG + SVM baseline defect detector.
3. Train and evaluate a YOLO11n object-detection model.
4. Investigate model optimisation through extended training and confidence-threshold sensitivity analysis.
5. Compare accuracy and computational performance.
6. Evaluate the final model on a held-out test dataset.

## 3. Dataset

The dataset consists of RGB images with bounding-box annotations originally provided in COCO format.

After dataset cleaning and preparation:

| Split | Images | Annotations |
|---|---:|---:|
| Training | 1,741 | 2,201 |
| Validation | 114 | 144 |
| Testing | 76 | 93 |
| Total | 1,931 | 2,438 |

Original-source filename overlaps were removed across dataset splits. However, visually similar frames may still be present because the dataset originates from video footage.

The COCO annotations were converted into normalised YOLO bounding-box labels for deep-learning training.

The dataset is excluded from the Git repository.

## 4. Computational Methods

### 4.1 HOG + SVM Baseline

The baseline uses HOG features to represent image regions and a linear SVM for binary defect classification.

A multiscale sliding-window approach is used to identify candidate defect locations. Non-maximum suppression removes overlapping predictions.

### 4.2 YOLO11n Object Detection

A pretrained YOLO11n model was fine-tuned on the pork packaging dataset.

Initial training settings:

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Image size | 640 |
| Batch size | 4 |
| Optimisation device | NVIDIA RTX 4050 Laptop GPU |
| Random seed | 42 |

An additional 20-epoch fine-tuning experiment was conducted using the best checkpoint from the initial training phase.

## 5. Model Evaluation

The models were evaluated using precision, recall, F1-score, mean Average Precision (mAP), and inference time.

### 5.1 HOG + SVM Versus Extended YOLO

The following results were obtained from 114 validation images using class-agnostic bounding-box matching at IoU 0.50.

| Metric | HOG + SVM | Extended YOLO |
|---|---:|---:|
| Precision | 0.12% | 84.62% |
| Recall | 0.69% | 53.47% |
| F1-score | 0.20% | 65.53% |
| Mean processing time | 4,362.98 ms | 33.40 ms |

The SVM implementation ran on the CPU, while YOLO used GPU acceleration.

### 5.2 Extended YOLO Validation Performance

| Metric | Result |
|---|---:|
| Precision | 84.40% |
| Recall | 59.74% |
| mAP@0.5 | 72.13% |
| mAP@0.5:0.95 | 34.00% |

### 5.3 Final Test Performance

| Metric | Result |
|---|---:|
| Precision | 47.44% |
| Recall | 51.64% |
| mAP@0.5 | 55.80% |
| mAP@0.5:0.95 | 25.50% |

The test performance indicates limitations in generalisation to unseen images.

## 6. Optimisation

The YOLO model underwent extended training and confidence-threshold sensitivity analysis.

Extended training increased validation mAP@0.5 from 60.29% to 72.13%.

Confidence thresholds between 0.10 and 0.60 were examined. A threshold of 0.30 was selected as a candidate operating point based on the observed precision–recall trade-off.

The final held-out test set was not used to select training parameters.

## 7. Software and Hardware

- Python 3.11.9
- Ultralytics YOLO
- PyTorch with CUDA
- OpenCV
- scikit-learn
- scikit-image
- NumPy
- pandas
- Matplotlib
- NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)

## 8. Repository Structure

```text
MMA3001-Pork-Packaging-Inspection/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_dataset_preparation.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_yolo_model.ipynb
├── src/
├── docs/
│   ├── index.md
│   ├── dataset.md
│   ├── methodology.md
│   ├── model_training.md
│   ├── evaluation.md
│   └── reproducibility.md
├── results/
│   └── figures/
├── mkdocs.yml
├── requirements.txt
├── README.md
└── .gitignore
```

The `data/` and `models/` directories are excluded from version control.

## 9. Running the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies using `pip install -r requirements.txt`.
4. Obtain the original annotated dataset and place it in the required `data/` folders.
5. Run the notebooks in numerical order, starting from dataset exploration.
6. Train the baseline and YOLO models, then evaluate their performance.

For GPU training, use a CUDA-compatible PyTorch installation and supported NVIDIA GPU.

## 10. Limitations

- The dataset contains substantial class imbalance.
- The wrinkle category has no labelled validation or test examples.
- Packaging-error detection remains challenging.
- Some images may be visually similar because they originate from video frames.
- The model cannot assess meat freshness or food safety.
- The final test performance is insufficient to support independent industrial deployment.

## 11. Conclusion

The project demonstrates that YOLO11n substantially outperforms a traditional HOG + SVM sliding-window baseline on the evaluated pork packaging dataset.

Extended training improved validation performance, but reduced test performance relative to validation highlights limitations in model generalisation.

Further improvements would require more diverse annotated images, greater representation of minority defect classes and additional evaluation under realistic industrial operating conditions.