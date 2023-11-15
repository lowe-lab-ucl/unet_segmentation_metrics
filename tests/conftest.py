import pytest
import numpy as np
import numpy.typing as npt
from typing import Tuple

SEED = 12345
RNG = np.random.default_rng(seed=SEED)


def _synthetic_image(sz: int = 32) -> npt.NDArray:
    image = np.zeros((sz, sz), dtype=np.uint8)
    boxsz = RNG.integers(low=sz // 4, high=sz - 1)
    xlo, ylo = RNG.integers(low=1, high=sz - boxsz, size=(2,))
    image[xlo : xlo + boxsz, ylo : ylo + boxsz] = 1
    return image


def _IoU(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    union = np.sum(np.logical_or(y_true, y_pred))
    intersection = np.sum(np.logical_and(y_true, y_pred))
    return intersection / union


@pytest.fixture
def image_pair() -> Tuple[npt.NDArray, npt.NDArray, float]:
    y_true = _synthetic_image()
    y_pred = _synthetic_image()
    return y_true, y_pred, _IoU(y_true, y_pred)
