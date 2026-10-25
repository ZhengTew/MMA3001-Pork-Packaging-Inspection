
# Reproducibility

## 1. Overview

This section describes the software environment, project
structure and execution procedure used to develop and
evaluate the pork packaging defect detection system.

The project implements two approaches:

- HOG + SVM traditional machine-learning baseline.
- YOLO11n deep-learning object detector.

The experiments are organised into four Jupyter notebooks.

Reproducing the results requires access to the original
dataset, the project source code and compatible Python
dependencies.

## 2. Software Environment

The original development environment used:

| Component | Configuration |
|---|---|
| Operating system | Windows |
| Python | 3.11.9 |
| Development environment | Visual Studio Code |
| Notebook environment | Jupyter |
| Deep learning framework | PyTorch |
| Object detection framework | Ultralytics |
| Image processing | OpenCV, scikit-image |
| Machine learning | scikit-learn |
| Data analysis | NumPy, pandas |
| Visualisation | Matplotlib |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| GPU memory | 6 GB |

The project dependencies are recorded in
`requirements.txt`.

Exact software versions and hardware configurations
may affect computational performance and numerical
reproducibility.

## 3. Project Structure

The main project directories are organised as follows:

    MMA3001-Pork-Packaging-Inspection/
    |
    |-- data/
    |   |-- processed/
    |   |-- yolo/
    |
    |-- docs/
    |   |-- index.md
    |   |-- dataset.md
    |   |-- methodology.md
    |   |-- model_training.md
    |   |-- evaluation.md
    |   |-- reproducibility.md
    |
    |-- models/
    |
    |-- notebooks/
    |   |-- 01_dataset_exploration.ipynb
    |   |-- 02_dataset_preparation.ipynb
    |   |-- 03_baseline_model.ipynb
    |   |-- 04_yolo_model.ipynb
    |
    |-- results/
    |
    |-- mkdocs.yml
    |-- requirements.txt
    |-- README.md

Large datasets and trained model checkpoints may be
excluded from Git version control.

Some output directories are generated automatically
when the corresponding notebooks are executed.

## 4. Installation

### 4.1 Clone the Repository

Clone the project from its GitHub repository.

    git clone <repository-url>

Replace `<repository-url>` with the actual repository URL.

Navigate to the project directory:

    cd MMA3001-Pork-Packaging-Inspection

### 4.2 Create a Virtual Environment

For Windows PowerShell:

    python -m venv .venv

Activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

If PowerShell blocks activation, the environment can
still be used by calling the Python executable directly.

### 4.3 Install Dependencies

Install the required packages:

    python -m pip install -r requirements.txt

The original experiments used a CUDA-enabled PyTorch
installation.

Users intending to reproduce GPU training should
verify that PyTorch recognises their NVIDIA GPU.

### 4.4 Verify GPU Availability

Run the following Python code:

    import torch

    print(torch.__version__)
    print(torch.cuda.is_available())

A CUDA-compatible environment should report `True`
for GPU availability.

CPU-only environments may require changes to the
YOLO device settings.

## 5. Dataset Requirements

The original image dataset is required to reproduce
dataset preparation and model training.

The dataset must contain COCO-format annotation files
and the corresponding images.

The processed dataset uses three splits:

| Split | Images | Annotations |
|---|---:|---:|
| Training | 1,741 | 2,201 |
| Validation | 114 | 144 |
| Testing | 76 | 93 |

The dataset preparation notebook produces the
processed annotations and split assignments.

The YOLO notebook converts these annotations
into the required YOLO format.

The YOLO dataset configuration is stored at:

`data/yolo/dataset.yaml`

The original dataset is not automatically downloaded
by the project notebooks.

## 6. Notebook Execution Order

The notebooks should be executed in numerical order.

### Notebook 01 — Dataset Exploration

File:

`notebooks/01_dataset_exploration.ipynb`

Purpose:

- Inspect the dataset structure.
- Calculate image and annotation counts.
- Examine defect class distributions.
- Visualise annotated images.
- Investigate duplicated source filenames.
- Identify potential data-quality issues.

### Notebook 02 — Dataset Preparation

File:

`notebooks/02_dataset_preparation.ipynb`

Purpose:

- Group images by original filename.
- Assign original-image groups to dataset splits.
- Remove cross-split filename overlap.
- Prepare cleaned COCO annotations.
- Validate image references and bounding boxes.
- Save the processed dataset.

This notebook modifies generated dataset files.

Review its file operations before rerunning it
on an existing processed dataset.

### Notebook 03 — HOG + SVM Baseline

File:

`notebooks/03_baseline_model.ipynb`

Purpose:

- Prepare positive and negative training samples.
- Extract HOG features.
- Train a linear SVM classifier.
- Evaluate cropped-region classification.
- Implement multiscale sliding-window detection.
- Apply non-maximum suppression.
- Evaluate full-image defect detection.

The trained SVM is saved as:

`models/hog_svm_baseline.joblib`

### Notebook 04 — YOLO Model

File:

`notebooks/04_yolo_model.ipynb`

Purpose:

- Convert COCO annotations into YOLO labels.
- Validate the converted dataset.
- Train the initial YOLO11n model.
- Continue training from the initial checkpoint.
- Compare validation performance.
- Analyse confidence thresholds.
- Benchmark runtime and GPU memory usage.
- Compare HOG + SVM and YOLO detection.
- Evaluate the selected model on test images.

The notebook uses checkpoints generated by
the training experiments.

## 7. Random Seeds

A random seed of 42 was used in the model
development experiments.

Random seeds were applied where supported
to reduce variation between experimental runs.

However, identical seeds do not guarantee
perfectly identical results.

Differences may arise from:

- GPU hardware.
- CUDA and PyTorch versions.
- Parallel processing.
- Nondeterministic operations.
- Dependency versions.

Reproduced results should therefore be
compared within a reasonable numerical tolerance.

## 8. Model Checkpoints

The HOG + SVM baseline is stored as a
serialised scikit-learn model.

The YOLO experiments generate trained
PyTorch checkpoint files.

The relevant YOLO checkpoints are:

- Initial training: `best.pt`.
- Extended training: `best.pt`.

The extended model uses the best checkpoint
from the initial experiment as its starting point.

Model checkpoint paths are established by
the training runs in Notebook 04.

If checkpoint files are unavailable,
the training experiments must be rerun.

Only load serialised models from trusted sources.

## 9. Evaluation Reproduction

### 9.1 HOG + SVM

The traditional baseline is evaluated on
validation images using:

- Sliding-window candidate regions.
- SVM decision threshold of 4.0.
- Non-maximum suppression at IoU 0.30.
- Detection matching at IoU 0.50.

### 9.2 YOLO

The extended YOLO model is evaluated using
the Ultralytics validation procedure.

The reported metrics include:

- Precision.
- Recall.
- mAP@0.5.
- mAP@0.5:0.95.

A confidence threshold of 0.30 is used
for the custom baseline-versus-YOLO comparison.

### 9.3 Detection Comparison

Both models are evaluated on the same
validation images.

Predictions are matched to ground-truth
annotations using class-agnostic,
one-to-one bounding-box matching.

The resulting true-positive, false-positive
and false-negative counts are used to
calculate precision, recall and F1-score.

This custom comparison is distinct from
Ultralytics' class-aware mAP evaluation.

## 10. Computational Benchmarking

YOLO inference benchmarks use:

| Parameter | Value |
|---|---|
| Image resolution | 640 |
| Confidence threshold | 0.30 |
| Batch size | 4 |
| GPU device | 0 |

GPU synchronisation is used before and
after timed inference operations.

The benchmark records:

- Mean processing time per image.
- Standard deviation of processing time.
- Throughput in frames per second.
- Peak allocated GPU memory.

Repeated measurements alternate model
execution order to reduce timing bias.

The recorded benchmark results are saved in:

`results/yolo_runtime_repeated.csv`

Runtime results depend on the hardware
and software environment.

## 11. Expected Results

### 11.1 Extended YOLO Validation

| Metric | Result |
|---|---:|
| Precision | 84.40% |
| Recall | 59.74% |
| mAP@0.5 | 72.13% |
| mAP@0.5:0.95 | 34.00% |

### 11.2 Final YOLO Test Evaluation

| Metric | Result |
|---|---:|
| Precision | 47.44% |
| Recall | 51.64% |
| mAP@0.5 | 55.80% |
| mAP@0.5:0.95 | 25.50% |

These values are reference results from
the original experiments.

Minor numerical differences may occur
when reproducing model training.

## 12. Known Limitations

Reproducibility is subject to several constraints.

### Dataset Availability

The original images and annotations
must be obtained separately if they
are not included in the repository.

### GPU Requirements

The original YOLO experiments used
CUDA GPU acceleration.

Different hardware may produce
different training times and results.

### Class Representation

The dataset contains uneven numbers
of examples across defect categories.

The wrinkle category is absent from
the validation and test annotations.

### Model Generalisation

The extended model achieved lower
performance on the held-out test
dataset than on the validation dataset.

Consequently, the reported results
should not be interpreted as proof
of reliable industrial performance.

### Data Independence

Original-filename grouping prevents
identical source filenames from
crossing dataset splits.

It does not guarantee that visually
similar video frames are independent.

## 13. Documentation Build

The project documentation is generated
using MkDocs.

Install the documentation tool:

    python -m pip install mkdocs

Preview the website locally:

    python -m mkdocs serve

Open the local address:

`http://127.0.0.1:8000/`

Generate the static HTML website:

    python -m mkdocs build

The HTML output is generated inside
the `site/` directory.

To check for documentation warnings:

    python -m mkdocs build --strict

## 14. Reproducibility Checklist

Before reproducing the experiments,
verify that:

- [ ] The original dataset is available.
- [ ] The Python environment is configured.
- [ ] Required dependencies are installed.
- [ ] The notebooks use the correct project paths.
- [ ] Dataset preparation completes successfully.
- [ ] COCO-to-YOLO conversion passes validation.
- [ ] The HOG + SVM classifier can be trained or loaded.
- [ ] YOLO checkpoint paths are available.
- [ ] Evaluation uses the documented thresholds.
- [ ] Results are compared against reference metrics.
- [ ] The MkDocs website builds without warnings.

## 15. Conclusion

The project provides a structured computational
workflow for automated pork packaging defect
detection.

The notebooks document dataset preparation,
traditional machine-learning development,
deep-learning model training and evaluation.

Combined with the documented environment
and experimental settings, these materials
support independent inspection and reproduction
of the computational methodology.
