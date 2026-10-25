
# Computational Methodology

## 1. Overview

This project investigates automated pork packaging defect
detection using two computer vision approaches:

1. Traditional machine learning using Histogram of Oriented
   Gradients (HOG) and Support Vector Machine (SVM).
2. Deep learning using YOLO11n object detection.

Both approaches were evaluated using bounding-box detection
metrics to compare their effectiveness.

## 2. Traditional Machine Learning: HOG + SVM

### 2.1 Image Preprocessing

Input images are processed using OpenCV.

Candidate image regions are converted to grayscale and
resized to 64 × 64 pixels before HOG feature extraction.

Consistent preprocessing is required during both model
training and inference.

### 2.2 HOG Feature Extraction

Histogram of Oriented Gradients describes local edge
orientations and image texture.

The following parameters were used:

| Parameter | Value |
|---|---|
| Input size | 64 × 64 pixels |
| Orientations | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalisation | L2-Hys |

Each image region is represented by a 1,764-element
feature vector.

### 2.3 SVM Classification

A linear Support Vector Machine was trained to distinguish
defect regions from background regions.

The classification pipeline included feature standardisation
using StandardScaler.

| Parameter | Value |
|---|---|
| Classifier | LinearSVC |
| C | 1.0 |
| Class weighting | Balanced |
| Maximum iterations | 10,000 |
| Random seed | 42 |

### 2.4 Sliding-Window Detection

For full-image detection, candidate regions are generated
using a multiscale sliding-window approach.

| Parameter | Value |
|---|---|
| Base window size | 128 × 128 pixels |
| Scale factors | 0.5, 1.0, 1.5 |
| Stride | 64 pixels |
| SVM decision threshold | 4.0 |
| NMS IoU threshold | 0.3 |

Each candidate region is converted into HOG features and
classified using the trained SVM.

Non-maximum suppression (NMS) removes overlapping
predictions.

The method can be computationally expensive because many
candidate regions must be evaluated individually.

## 3. Deep Learning: YOLO11n

### 3.1 Model Selection

YOLO11n was selected as a lightweight object-detection
architecture.

Unlike the sliding-window baseline, YOLO predicts object
locations and classes directly from an input image.

The pretrained `yolo11n.pt` checkpoint was used for
transfer learning.

### 3.2 Dataset Format

YOLO requires images and corresponding normalised
bounding-box annotations.

The processed COCO dataset was converted into YOLO format
with five defect classes.

The dataset configuration is stored in:

`data/yolo/dataset.yaml`

### 3.3 Initial Training

The initial model was trained using the following settings:

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Epochs | 20 |
| Image resolution | 640 |
| Batch size | 4 |
| Device | CUDA GPU |
| Random seed | 42 |

Training was performed using the Ultralytics framework.

### 3.4 Extended Training

An additional training experiment was performed using
the best checkpoint from the initial training run.

The extended experiment used another 20 training epochs
to investigate whether additional optimisation improved
detection performance.

The best checkpoint from the extended experiment was
used for subsequent evaluation.

## 4. Detection Postprocessing

### 4.1 Confidence Thresholding

YOLO predictions are filtered using a confidence threshold.

A confidence-threshold sensitivity analysis was conducted
to investigate changes in precision and recall.

A confidence threshold of 0.30 was used for the subsequent
custom model comparison.

### 4.2 Non-Maximum Suppression

Non-maximum suppression removes redundant overlapping
predictions.

Its IoU threshold is distinct from the IoU threshold used
to determine whether a prediction matches a ground-truth box.

## 5. Evaluation Methodology

### 5.1 Intersection over Union

Intersection over Union (IoU) measures the overlap between
a predicted bounding box and its ground-truth box.

IoU = Intersection Area / Union Area

A prediction is considered correctly localised when
its IoU meets the evaluation threshold.

### 5.2 Detection Metrics

The following metrics are used:

**Precision**

Precision = TP / (TP + FP)

**Recall**

Recall = TP / (TP + FN)

**F1-score**

F1 = 2 × Precision × Recall / (Precision + Recall)

Where:

- TP: True positives
- FP: False positives
- FN: False negatives

YOLO is also evaluated using mAP@0.5 and mAP@0.5:0.95.

### 5.3 Baseline-versus-YOLO Comparison

Both detectors are compared on the same validation images.

Predicted and ground-truth bounding boxes are matched
using an IoU threshold of 0.50.

Each ground-truth box can be matched to a maximum of one
prediction.

This custom comparison is class-agnostic and evaluates
defect localisation rather than class identification.

The comparison uses each model's specified detection
threshold, so the results represent performance at those
operating points.

## 6. Computational Performance

Inference time was measured using Python's
`time.perf_counter()`.

For GPU inference, CUDA synchronisation was performed
before and after timing to account for asynchronous
GPU operations.

The following computational metrics were collected:

- Mean processing time per image
- Standard deviation of processing time
- Throughput in frames per second
- Peak PyTorch GPU memory allocation

Repeated benchmarking was performed with alternating
model evaluation order to reduce measurement-order bias.

## 7. Methodological Limitations

### HOG + SVM

- Detection depends on predefined window sizes and strides.
- Feature extraction is computationally expensive.
- The model does not directly predict defect categories.
- Performance depends on feature extraction and thresholding.

### YOLO11n

- Detection performance depends on the quality and
  representativeness of training annotations.
- Underrepresented classes may be detected less reliably.
- Confidence thresholds affect precision and recall.
- GPU performance depends on the available hardware.

### Comparison Limitations

HOG + SVM inference was performed on the CPU, while
YOLO inference was performed on a CUDA GPU.

Therefore, runtime measurements compare the implemented
systems rather than identical hardware configurations.

The class-agnostic comparison also does not establish
whether YOLO correctly identifies each defect category.

## 8. Implementation

The computational methods are implemented in:

- `notebooks/03_baseline_model.ipynb`
- `notebooks/04_yolo_model.ipynb`

The notebooks provide training code, experimental
parameters, evaluation procedures and visualisations.
