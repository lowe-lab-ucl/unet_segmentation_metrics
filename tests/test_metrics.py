import pytest
import numpy as np

import umetrix


STRICT_PARAMS = [(False, 0.0), (True, 0.1), (True, 0.2), (True, 0.5), (True, 0.7)]


@pytest.mark.parametrize("strict,iou_threshold", STRICT_PARAMS)
def test_calculate(image_pair, strict, iou_threshold):
    """Run the metrics on a pair of images."""
    y_true, y_pred, stats = image_pair
    IoU = stats["IoU"]

    result = umetrix.calculate(
        y_true, y_pred, strict=strict, iou_threshold=iou_threshold
    )

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

    result = umetrix.calculate(y_true, y_pred)
    assert result.n_true_labels == 0
    assert result.n_pred_labels == 1
    assert result.n_true_positives == 0
    assert result.n_false_negatives == 0
    assert result.n_false_positives == 1


def test_calculate_no_pred(image_pair):
    """Test a pair of images where there is no object in the prediction."""
    y_true, y_pred, _ = image_pair
    y_pred = np.zeros_like(y_true)

    result = umetrix.calculate(y_true, y_pred)
    assert result.n_true_labels == 1
    assert result.n_pred_labels == 0
    assert result.n_true_positives == 0
    assert result.n_false_negatives == 1
    assert result.n_false_positives == 0


@pytest.mark.parametrize("strict,iou_threshold", STRICT_PARAMS)
def test_calculate_grid(image_grid, strict, iou_threshold):
    """Test a multi-instance segmentation."""
    y_true, y_pred, stats = image_grid
    result = umetrix.calculate(
        y_true, y_pred, strict=strict, iou_threshold=iou_threshold
    )

    n_iou_over_threshold = sum([iou > iou_threshold for iou in stats["IoU"]])
    n_iou_under_threshold = stats["n_pairs"] - n_iou_over_threshold if strict else 0
    n_tp = n_iou_over_threshold
    n_fp = stats["n_false_positive"] + n_iou_under_threshold
    n_fn = stats["n_false_negative"] + n_iou_under_threshold

    assert result.n_true_labels == stats["n_true"]
    assert result.n_pred_labels == stats["n_pred"]
    assert result.n_true_positives == n_tp
    assert result.n_false_positives == n_fp
    assert result.n_false_negatives == n_fn


@pytest.mark.parametrize("strict,iou_threshold", STRICT_PARAMS)
def test_real(real_image_pair, strict, iou_threshold):
    """Test a real image pair."""
    y_true, y_pred = real_image_pair
    result = umetrix.calculate(
        y_true, y_pred, strict=strict, iou_threshold=iou_threshold
    )
    assert result.n_true_labels == 13
    assert result.n_pred_labels == 14
