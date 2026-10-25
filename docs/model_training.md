
# Model Training and Optimisation

## 1. Overview

This project developed a YOLO11n object detector to identify
five categories of pork packaging and visual meat defects.

The training process consisted of two experiments:

1. Initial training using pretrained YOLO11n weights.
2. Extended training using the best checkpoint from the
   initial experiment.

The objective was to evaluate whether additional training
improved detection performance.

## 2. Training Environment

Model training was performed using Python and the
Ultralytics YOLO framework.

| Component | Configuration |
|---|---|
| Python | 3.11.9 |
| Model architecture | YOLO11n |
| Deep learning framework | PyTorch |
| Training library | Ultralytics |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| GPU memory | 6 GB |
| Acceleration | CUDA |
| Random seed | 42 |

The GPU was used to accelerate training and inference.

The recorded software versions and dependencies are
documented in the project's requirements file.

## 3. Training Dataset

The model was trained using the processed dataset
following duplicate-filename checks.

| Split | Images | Annotations |
|---|---:|---:|
| Training | 1,741 | 2,201 |
| Validation | 114 | 144 |
| Testing | 76 | 93 |

The training dataset was used for parameter optimisation.

The validation dataset was used to assess performance
during model development.

The test dataset was reserved for final evaluation.

All images and labels were organised in the directory:

`data/yolo/`

The dataset configuration was defined in:

`data/yolo/dataset.yaml`

## 4. Initial YOLO Training

### 4.1 Model Initialisation

The initial experiment used the pretrained checkpoint:

`yolo11n.pt`

Pretrained weights provided a starting point for
fine-tuning the detector on the pork packaging dataset.

### 4.2 Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Image resolution | 640 × 640 |
| Batch size | 4 |
| GPU device | 0 |
| Data-loader workers | 0 |
| Random seed | 42 |
| Training run | initial_training |

Training outputs, including model checkpoints and
performance figures, were saved by Ultralytics.

The best checkpoint was stored as `best.pt` within
the experiment's weights directory.

### 4.3 Initial Validation Results

| Metric | Initial YOLO |
|---|---:|
| Precision | 64.63% |
| Recall | 55.59% |
| mAP@0.5 | 60.29% |
| mAP@0.5:0.95 | 29.14% |

The initial model demonstrated the ability to detect
packaging-related defects.

However, recall and bounding-box localisation accuracy
indicated potential for further improvement.

## 5. Extended YOLO Training

### 5.1 Training Approach

The extended experiment was initialised using the best
checkpoint from the initial experiment.

An additional 20 training epochs were performed using
the same processed dataset.

This continued optimisation from previously learned
model parameters rather than starting again from
the original pretrained checkpoint.

### 5.2 Extended Training Configuration

| Parameter | Value |
|---|---|
| Starting weights | Initial best.pt |
| Additional epochs | 20 |
| Image resolution | 640 × 640 |
| Batch size | 4 |
| GPU device | 0 |
| Data-loader workers | 0 |
| Random seed | 42 |
| Training run | extended_training |

The best checkpoint from this experiment was selected
as the final model for subsequent evaluation.

## 6. Validation Performance Comparison

| Metric | Initial YOLO | Extended YOLO | Improvement |
|---|---:|---:|---:|
| Precision | 64.63% | 84.40% | +19.77 pp |
| Recall | 55.59% | 59.74% | +4.15 pp |
| mAP@0.5 | 60.29% | 72.13% | +11.84 pp |
| mAP@0.5:0.95 | 29.14% | 34.00% | +4.86 pp |

*pp = percentage points.*

Extended training improved all four reported validation
metrics.

The largest improvement was observed in precision,
which increased by 19.77 percentage points.

The mAP@0.5 improvement indicates better detection
performance across confidence thresholds.

The improvement in mAP@0.5:0.95 was smaller, suggesting
that accurate bounding-box localisation remained challenging.

These results supported selecting the extended YOLO
checkpoint for further evaluation.

## 7. Confidence-Threshold Sensitivity

The extended model was evaluated using several
confidence thresholds.

| Confidence | Precision | Recall |
|---|---:|---:|
| 0.10 | 84.40% | 59.74% |
| 0.20 | 84.40% | 59.74% |
| 0.30 | 83.20% | 60.18% |
| 0.40 | 93.47% | 51.46% |
| 0.50 | 90.04% | 29.40% |
| 0.60 | 74.29% | 16.48% |

### 7.1 Interpretation

The results demonstrate the effect of confidence
threshold selection on detection performance.

A threshold of 0.30 produced the highest observed
recall among the tested values.

A threshold of 0.40 produced the highest observed
precision.

Higher thresholds generally reduced the number of
retained detections, although precision did not
increase monotonically.

### 7.2 Selected Operating Threshold

A confidence threshold of 0.30 was selected for
the custom comparison with the HOG + SVM baseline.

This threshold provides a useful operating point for
examining the balance between correctly detected
defects and missed defects.

The selected threshold does not necessarily represent
an optimal industrial operating point.

### 7.3 Evaluation Considerations

The confidence-threshold experiment used an NMS
IoU threshold of 0.70.

This is separate from the IoU thresholds used
to calculate object-detection evaluation metrics.

The mAP values produced under different confidence
cutoffs should not be interpreted as directly
comparable standard full-confidence-range AP values.

## 8. Model Selection

The extended YOLO11n checkpoint was selected for
final evaluation because it achieved better validation
performance than the initial model.

The final model was then evaluated on the held-out
test dataset.

The test results are reported separately in the
Evaluation and Results documentation.

This separation is important because validation
performance may not accurately represent performance
on previously unseen images.

## 9. Training Limitations

### Class Imbalance

Some defect classes contain substantially fewer
training examples than others.

This can affect the model's ability to learn
reliable features for underrepresented categories.

### Missing Evaluation Classes

The wrinkle category has no annotations in the
validation or test datasets.

Consequently, its detection performance cannot be
reliably established from these evaluation splits.

### Limited Dataset Diversity

Images from similar capture conditions may share
visual characteristics.

The trained model may therefore perform less
reliably on substantially different packaging,
lighting or camera conditions.

### Limited Hyperparameter Exploration

The project investigated additional training
and confidence-threshold sensitivity.

It did not perform an exhaustive search across
architectures or training hyperparameters.

## 10. Reproducibility

The training and optimisation experiments are
implemented in:

`notebooks/04_yolo_model.ipynb`

The notebook includes:

- YOLO dataset conversion and validation.
- Initial model training.
- Extended model training.
- Validation metric extraction.
- Confidence-threshold sensitivity analysis.
- Runtime and GPU memory benchmarking.
- Final model evaluation.

Training checkpoints and large datasets may need to
be obtained or regenerated separately if they are
excluded from the GitHub repository.
