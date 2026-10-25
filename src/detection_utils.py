
"""Utilities for bounding-box matching and detection evaluation."""


def box_iou(box_a, box_b):
    """Calculate IoU between boxes in [x, y, width, height] format.

    Args:
        box_a (sequence): First bounding box.
        box_b (sequence): Second bounding box.

    Returns:
        float: Intersection over Union between 0 and 1.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection

    return intersection / union if union > 0 else 0.0


def match_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predictions to ground truth using one-to-one IoU matching.

    Args:
        pred_boxes (list): Predicted boxes ordered by descending score.
        gt_boxes (list): Ground-truth boxes.
        iou_threshold (float): Minimum matching IoU.

    Returns:
        tuple: True positives, false positives, and false negatives.
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for predicted_box in pred_boxes:
        best_iou = 0
        best_index = None

        for index, gt_box in enumerate(gt_boxes):
            if index in matched_gt:
                continue

            iou = box_iou(predicted_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_index)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


def calculate_detection_metrics(tp, fp, fn):
    """Calculate precision, recall, and F1-score.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        tuple: Precision, recall, and F1-score.
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return precision, recall, f1
