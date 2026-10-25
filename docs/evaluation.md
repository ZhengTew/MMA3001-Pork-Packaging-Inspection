
# Evaluation and Results

## 1. Overview

This project evaluates two approaches to automated
pork packaging defect detection:

1. HOG + SVM sliding-window detection.
2. YOLO11n deep-learning object detection.

Evaluation considers detection accuracy, computational
performance and generalisation to unseen images.

The experiments include:

- HOG + SVM baseline evaluation.
- Initial and extended YOLO validation.
- Comparison of detection performance.
- Inference time and GPU memory benchmarking.
- Final evaluation on the held-out test dataset.

## 2. Evaluation Dataset

The processed dataset contains the following splits:

| Split | Images | Annotations |
|---|---:|---:|
| Training | 1,741 | 2,201 |
| Validation | 114 | 144 |
| Testing | 76 | 93 |

The validation dataset was used for model development
and performance comparison.

The test dataset was reserved for the final assessment
of the selected YOLO model.

## 3. Evaluation Metrics

### 3.1 True and False Detections

Detections were evaluated by matching predicted
bounding boxes to ground-truth annotations.

For the custom model comparison, an IoU threshold
of 0.50 was used.

Each ground-truth box could be matched to only
one prediction.

The resulting counts were:

- True Positive (TP): Correctly localised defect.
- False Positive (FP): Unmatched predicted defect.
- False Negative (FN): Missed annotated defect.

### 3.2 Precision

Precision measures the proportion of predictions
that correspond to ground-truth objects.

Precision = TP / (TP + FP)

### 3.3 Recall

Recall measures the proportion of ground-truth
objects successfully detected.

Recall = TP / (TP + FN)

### 3.4 F1-score

The F1-score combines precision and recall.

F1 = 2 × Precision × Recall / (Precision + Recall)

### 3.5 Mean Average Precision

YOLO performance was also evaluated using:

- mAP@0.5: Mean average precision at IoU 0.50.
- mAP@0.5:0.95: Mean average precision averaged
  across IoU thresholds from 0.50 to 0.95.

These metrics assess object detection across
confidence thresholds and, for mAP@0.5:0.95,
multiple localisation requirements.

## 4. HOG + SVM Baseline Results

The traditional baseline used HOG features
with a linear Support Vector Machine.

Defects were detected using multiscale
sliding windows and non-maximum suppression.

### 4.1 Full-Image Detection Results

The baseline was evaluated on 114 validation
images containing 144 annotated defects.

| Metric | Result |
|---|---:|
| True Positives | 1 |
| False Positives | 832 |
| False Negatives | 143 |
| Precision | 0.12% |
| Recall | 0.69% |
| F1-score | 0.20% |
| Mean processing time | 4,362.98 ms/image |

### 4.2 Interpretation

The baseline achieved very limited
full-image detection performance.

Only one annotated defect was successfully
matched at IoU 0.50.

The large number of false positives indicates
that many background regions were incorrectly
identified as defects.

The low recall shows that most annotated
defects were not detected.

Although the SVM achieved approximately 85%
classification accuracy during cropped-region
evaluation, this did not translate into
effective full-image object detection.

This distinction illustrates the additional
difficulty of defect localisation compared
with classifying predefined image regions.

## 5. YOLO Validation Results

### 5.1 Initial and Extended Models

| Metric | Initial YOLO | Extended YOLO |
|---|---:|---:|
| Precision | 64.63% | 84.40% |
| Recall | 55.59% | 59.74% |
| mAP@0.5 | 60.29% | 72.13% |
| mAP@0.5:0.95 | 29.14% | 34.00% |

### 5.2 Improvements After Extended Training

| Metric | Improvement |
|---|---:|
| Precision | +19.77 percentage points |
| Recall | +4.15 percentage points |
| mAP@0.5 | +11.84 percentage points |
| mAP@0.5:0.95 | +4.86 percentage points |

Extended training improved all four
reported validation metrics.

The largest increase was observed in precision.

The increase in mAP@0.5 indicates improved
overall detection performance.

However, mAP@0.5:0.95 remained relatively low,
suggesting that accurate bounding-box
localisation continued to be challenging.

## 6. HOG + SVM Versus YOLO

A custom evaluation procedure was developed
to compare both approaches using identical
validation images.

The comparison used:

- One-to-one bounding-box matching.
- An IoU threshold of 0.50.
- Model-specific detection thresholds.
- Class-agnostic matching.

The HOG + SVM baseline produced very low
precision and recall during full-image
detection.

In contrast, YOLO achieved substantially
stronger validation detection performance.

However, the reported Ultralytics validation
metrics and custom matching metrics are
calculated differently.

Therefore, results from these separate
evaluation procedures should not be treated
as directly interchangeable.

The custom comparison implemented in
`04_yolo_model.ipynb` provides the
more consistent basis for direct comparison
at fixed detection thresholds.

## 7. Final YOLO Test Results

The extended YOLO model was evaluated on
the held-out test dataset.

The dataset contained 76 images and
93 annotated defects.

| Metric | Test Result |
|---|---:|
| Precision | 47.44% |
| Recall | 51.64% |
| mAP@0.5 | 55.80% |
| mAP@0.5:0.95 | 25.50% |

### 7.1 Interpretation

The extended model successfully detected
some defects in previously unseen images.

However, test performance was lower than
the reported validation performance.

This indicates that the validation results
did not fully represent model performance
on the held-out test dataset.

Possible contributing factors include:

- Limited dataset size.
- Unequal class representation.
- Differences in image conditions.
- Similarity among training images.
- Difficulty distinguishing subtle defects.

These explanations are plausible limitations
rather than individually verified causes
of the performance reduction.

### 7.2 Generalisation Gap

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 84.40% | 47.44% |
| Recall | 59.74% | 51.64% |
| mAP@0.5 | 72.13% | 55.80% |
| mAP@0.5:0.95 | 34.00% | 25.50% |

The decrease in test performance demonstrates
the importance of evaluating detection models
on held-out data.

The available results are insufficient to
establish reliable performance under different
industrial operating conditions.

## 8. Computational Performance

### 8.1 Inference Benchmarking

The initial and extended YOLO models were
benchmarked using identical test images.

The benchmarking procedure measured:

- Mean time per image.
- Standard deviation of processing time.
- Throughput in frames per second.
- Peak allocated GPU memory.

Both models used:

| Parameter | Value |
|---|---|
| Image resolution | 640 |
| Confidence threshold | 0.30 |
| Batch size | 4 |
| GPU device | 0 |

GPU synchronisation was used to improve
timing accuracy.

### 8.2 Repeated Benchmarking

Additional benchmarking was conducted over
three rounds.

Model evaluation order was alternated
between rounds to reduce ordering effects.

Each round used two internal timing
repetitions per model.

The recorded results were exported to:

`results/yolo_runtime_repeated.csv`

Average runtime and throughput were
calculated to examine performance consistency.

### 8.3 Interpretation

The lightweight YOLO11n architecture
supports efficient GPU-based inference.

However, computational comparisons must
account for hardware differences.

The traditional baseline was evaluated
on the CPU, whereas YOLO was evaluated
on an NVIDIA GPU.

Consequently, the timing results describe
the implemented systems rather than a
controlled comparison on identical hardware.

Measured processing time also includes
model preprocessing and postprocessing.

## 9. Qualitative Evaluation

Ground-truth annotations were visually
compared with model predictions.

The qualitative analysis considered:

- Correctly detected defects.
- Missed annotated defects.
- False-positive detections.
- Bounding-box localisation quality.
- Differences across defect categories.

These comparisons help identify failure
modes that may not be obvious from
aggregate numerical metrics.

## 10. Limitations

### 10.1 Dataset Imbalance

The five defect categories are not
equally represented.

The wrinkle category has no annotated
examples in the validation or test sets.

Its detection performance therefore
cannot be assessed reliably using
these evaluation splits.

### 10.2 Detection Thresholds

Precision and recall depend on the
confidence or decision threshold.

A threshold suitable for one
application may not be appropriate
for another.

### 10.3 Class-Agnostic Comparison

The custom HOG + SVM versus YOLO
comparison evaluates bounding-box
localisation without requiring
correct class identification.

It does not measure multiclass
classification performance.

### 10.4 Hardware Differences

CPU and GPU runtime measurements
should not be interpreted as
equal-hardware performance.

### 10.5 Limited Test Coverage

The held-out test dataset contains
76 images.

A larger and more diverse test
dataset would provide stronger
evidence of generalisation.

## 11. Overall Findings

The experiments show that:

1. HOG + SVM performed poorly when
   applied to full-image defect detection.

2. YOLO11n achieved substantially
   better detection performance during
   validation.

3. Extended training improved the
   reported YOLO validation metrics.

4. Performance decreased on the
   held-out test dataset.

5. Additional data and model refinement
   are necessary before considering
   real-world deployment.

The results demonstrate the potential
of deep-learning object detection for
packaging inspection while identifying
important limitations in robustness
and generalisation.

## 12. Supporting Implementation

Evaluation procedures and figures
are implemented in:

- `notebooks/03_baseline_model.ipynb`
- `notebooks/04_yolo_model.ipynb`

Additional runtime measurements
are stored in the `results/` directory.
