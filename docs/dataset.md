
# Dataset Preparation

## 1. Dataset Overview

The project uses RGB images of pork rasher products with
bounding-box annotations for packaging and visual meat
presentation defects.

The original dataset was exported in COCO JSON format and
contains five annotated defect categories:

| COCO ID | Defect Category |
|---|---|
| 1 | Loose meat |
| 2 | Packaging error |
| 3 | Twisted meat |
| 4 | Unsealed |
| 5 | Wrinkle |

COCO category ID 0 is an unused placeholder and is excluded
from the YOLO class mapping.

## 2. Original Dataset

The original dataset contains 1,955 images and 2,466
bounding-box annotations.

| Dataset | Images | Annotations |
|---|---:|---:|
| Training | 1,764 | 2,232 |
| Validation | 115 | 143 |
| Testing | 76 | 91 |
| **Total** | **1,955** | **2,466** |

Images and annotations were inspected using Python, OpenCV,
pandas and Matplotlib.

## 3. Dataset Quality Checks

The dataset exploration included:

- Counting images and annotations in each split.
- Examining class distributions.
- Displaying images with ground-truth bounding boxes.
- Checking for invalid annotation geometry.
- Identifying duplicated original filenames across splits.

No invalid bounding boxes were identified by the implemented
geometry checks.

However, original-image filename overlaps were identified
between training, validation and testing datasets.

Such overlaps can cause data leakage and produce overly
optimistic evaluation results.

## 4. Duplicate Handling

Images were grouped by their original source filename.

Each original-image group was assigned to only one dataset
split using the following priority:

1. Test
2. Validation
3. Training

This approach prevents copies with the same original filename
from appearing across different splits.

For validation and testing, one exported copy per original
image was retained. Training copies were preserved.

The resulting dataset was saved separately under
`data/processed/` to avoid modifying the original files.

## 5. Processed Dataset

| Dataset | Images | Annotations |
|---|---:|---:|
| Training | 1,741 | 2,201 |
| Validation | 114 | 144 |
| Testing | 76 | 93 |
| **Total** | **1,931** | **2,438** |

The processed dataset was verified by checking:

- Existence of all referenced image files.
- Valid image and category references.
- Positive bounding-box dimensions.
- Bounding boxes within image boundaries.
- No cross-split original-filename overlaps.

All implemented validation checks passed.

## 6. COCO-to-YOLO Conversion

The processed COCO annotations were converted to YOLO
format for object-detection training.

COCO bounding boxes use:

`[x_min, y_min, width, height]`

YOLO labels use:

`[class_id, x_center, y_center, width, height]`

The YOLO box coordinates are normalised by the corresponding
image width and height.

The five classes were mapped to consecutive YOLO IDs:

| YOLO ID | Defect Category |
|---|---|
| 0 | Loose meat |
| 1 | Packaging error |
| 2 | Twisted meat |
| 3 | Unsealed |
| 4 | Wrinkle |

Each image has a corresponding text label file. Images without
annotations have empty label files.

The converted dataset is stored under `data/yolo/`.

## 7. Dataset Limitations

### Class Imbalance

The dataset contains unequal numbers of examples across
defect categories.

This may reduce detection performance for less-represented
classes.

### Wrinkle Category

The processed training set contains wrinkle annotations,
but the validation and test datasets contain none.

Therefore, the model's ability to detect wrinkle defects
cannot be reliably assessed using the available evaluation
datasets.

### Video-Frame Similarity

Some images originate from nearby video frames.

Although original-filename overlaps were removed, visually
similar frames may still occur across dataset splits.

Consequently, filename-based separation does not guarantee
complete statistical independence.

### Annotation Quality

Geometric validation cannot establish whether the defect
categories or bounding-box locations are visually correct.

Further manual annotation review would improve confidence
in the dataset.

## 8. Reproducibility

Dataset exploration and preparation are implemented in:

- `notebooks/01_dataset_exploration.ipynb`
- `notebooks/02_dataset_preparation.ipynb`

COCO-to-YOLO conversion is implemented in:

- `notebooks/04_yolo_model.ipynb`

The original and processed image datasets are excluded from
Git version control.
