import pytest
import numpy as np

import umetrics


@pytest.mark.parametrize("strict", (False, True))
def test_calculate(image_pair, strict):
    """Run the metrics on a pair of images."""
    y_true, y_pred, stats = image_pair
    IoU = stats["IoU"]

    result = umetrics.calculate(y_true, y_pred, strict=strict)

    # calculate the real number of true postives based on strict matching
    real_tp = int(IoU > result.iou_threshold) if strict else int(IoU > 0)

    assert result.n_true_labels == 1
    assert result.n_pred_labels == 1
    assert result.n_true_positives == real_tp
    assert result.n_false_positives == 1 - real_tp


def test_calculate_no_true(image_pair):
    """Test a pair of images where there is no object in the GT."""
    y_true, y_pred, _ = image_pair
    y_true = np.zeros_like(y_pred)

    result = umetrics.calculate(y_true, y_pred)
    assert result.n_true_labels == 0
    assert result.n_pred_labels == 1
    assert result.n_true_positives == 0
    assert result.n_false_negatives == 0
    assert result.n_false_positives == 1


def test_calculate_no_pred(image_pair):
    """Test a pair of images where there is no object in the prediction."""
    y_true, y_pred, _ = image_pair
    y_pred = np.zeros_like(y_true)

    result = umetrics.calculate(y_true, y_pred)
    assert result.n_true_labels == 1
    assert result.n_pred_labels == 0
    assert result.n_true_positives == 0
    assert result.n_false_negatives == 1
    assert result.n_false_positives == 0


def test_calculate_grid(image_grid):
    """Test a multi-instance segmentation."""
    y_true, y_pred, stats = image_grid
    result = umetrics.calculate(y_true, y_pred)

    assert result.n_true_labels == stats["n_true"]
    assert result.n_pred_labels == stats["n_pred"]
    assert result.n_true_positives == stats["n_pairs"]
    assert result.n_false_positives == stats["n_missing_true"]
    assert result.n_false_negatives == stats["n_missing_pred"]
