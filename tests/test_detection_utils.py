
"""Unit tests for bounding-box matching and detection metrics."""

import pytest

from src.detection_utils import (
    box_iou,
    match_detections,
    calculate_detection_metrics,
)


def test_iou_identical_boxes():
    """Identical boxes should have an IoU of 1."""
    box = [10, 10, 20, 20]
    assert box_iou(box, box) == pytest.approx(1.0)


def test_iou_non_overlapping_boxes():
    """Separate boxes should have zero overlap."""
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 10, 10]

    assert box_iou(box_a, box_b) == 0.0


def test_iou_partial_overlap():
    """Partially overlapping boxes should have the expected IoU."""
    box_a = [0, 0, 10, 10]
    box_b = [5, 0, 10, 10]

    assert box_iou(box_a, box_b) == pytest.approx(1 / 3)


def test_iou_zero_area():
    """A zero-area box should return zero IoU."""
    assert box_iou([0, 0, 0, 10], [0, 0, 10, 10]) == 0.0


def test_matching_correct_detection():
    """One correctly localised prediction should be a true positive."""
    predictions = [[10, 10, 20, 20]]
    ground_truth = [[10, 10, 20, 20]]

    assert match_detections(predictions, ground_truth) == (1, 0, 0)


def test_matching_duplicate_predictions():
    """Multiple predictions cannot match the same ground-truth box."""
    predictions = [[10, 10, 20, 20], [10, 10, 20, 20]]
    ground_truth = [[10, 10, 20, 20]]

    assert match_detections(predictions, ground_truth) == (1, 1, 0)


def test_matching_missed_defect():
    """An unmatched ground-truth box should count as a false negative."""
    assert match_detections([], [[10, 10, 20, 20]]) == (0, 0, 1)


def test_matching_no_ground_truth():
    """A prediction without ground truth should count as a false positive."""
    assert match_detections([[10, 10, 20, 20]], []) == (0, 1, 0)


def test_detection_metrics():
    """Precision, recall, and F1 should match independently known values."""
    precision, recall, f1 = calculate_detection_metrics(8, 2, 4)

    assert precision == pytest.approx(0.8)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(8 / 11)


def test_detection_metrics_zero_counts():
    """Empty detection counts should not cause division by zero."""
    assert calculate_detection_metrics(0, 0, 0) == (0, 0, 0)
