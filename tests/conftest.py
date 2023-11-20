import pytest
import numpy as np
import numpy.typing as npt

from pathlib import Path
from skimage.io import imread
from skimage.util import montage
from typing import Tuple

from umetrix.core import IoU

SEED = 12343
RNG = np.random.default_rng(seed=SEED)


def _synthetic_image(sz: int = 32) -> npt.NDArray:
    image = np.zeros((sz, sz), dtype=np.uint8)
    boxsz = RNG.integers(low=sz // 4, high=sz - 1)
    xlo, ylo = RNG.integers(low=1, high=sz - boxsz, size=(2,))
    image[xlo : xlo + boxsz, ylo : ylo + boxsz] = 1
    return image


# def _IoU(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
#     union = np.sum(np.logical_or(y_true, y_pred))
#     intersection = np.sum(np.logical_and(y_true, y_pred))
#     return intersection / union


@pytest.fixture
def image_grid(N: int = 3, sz: int = 32) -> Tuple[npt.NDArray, npt.NDArray, dict]:
    image_types = RNG.choice(
        ["pair", "missing_true", "missing_pred"], size=(N * N,)
    ).tolist()
    true_stack = np.zeros((N * N, sz, sz), dtype=np.uint8)
    pred_stack = np.zeros((N * N, sz, sz), dtype=np.uint8)

    ious = []

    n_true_positive = 0
    n_false_positive = 0
    n_false_negative = 0

    for idx, img_type in enumerate(image_types):
        if img_type == "pair":
            true_stack[idx, ...] = _synthetic_image()
            pred_stack[idx, ...] = _synthetic_image()
            iou = IoU(true_stack[idx, ...], pred_stack[idx, ...])
            ious.append(iou)
            if iou > 0:
                n_true_positive += 1
            else:
                n_false_positive += 1
                n_false_negative += 1
        elif img_type == "missing_true":
            pred_stack[idx, ...] = _synthetic_image()
            ious.append(0.0)
            n_false_positive += 1
        else:
            true_stack[idx, ...] = _synthetic_image()
            ious.append(0.0)
            n_false_negative += 1

    # number of pairs where there is some overlap
    n_pairs = image_types.count("pair")
    n_missing_true = image_types.count("missing_true")
    n_missing_pred = image_types.count("missing_pred")

    stats = {
        "n_pairs": n_pairs,
        "n_true": n_pairs + n_missing_pred,
        "n_pred": n_pairs + n_missing_true,
        "n_true_positive": n_true_positive,
        "n_false_positive": n_false_positive,
        "n_false_negative": n_false_negative,
        "n_total": len(image_types),
        "IoU": ious,
    }

    return (
        montage(true_stack, rescale_intensity=False, grid_shape=(N, N)),
        montage(pred_stack, rescale_intensity=False, grid_shape=(N, N)),
        stats,
    )


@pytest.fixture
def image_pair() -> Tuple[npt.NDArray, npt.NDArray, dict]:
    y_true = _synthetic_image()
    y_pred = _synthetic_image()
    stats = {"IoU": IoU(y_true, y_pred)}
    return y_true, y_pred, stats


@pytest.fixture
def real_image_pair() -> Tuple[npt.NDArray, npt.NDArray]:
    filename = Path(__file__).parent.resolve() / "data" / "unet.tif"
    img = (imread(filename) > 0).astype(np.uint8)
    return img[0, ...], img[1, ...]
