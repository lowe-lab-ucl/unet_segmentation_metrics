from __future__ import annotations

import enum
import numpy as np
import numpy.typing as npt

from skimage.io import imread
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from scipy.ndimage import find_objects
from scipy.optimize import linear_sum_assignment

from typing import Dict

from . import render


class Metrics(str, enum.Enum):
    N_TRUE_LABELS = "n_true_labels"
    N_PRED_LABELS = "n_pred_labels"
    N_TRUE_POSITIVES = "n_true_positives"
    N_FALSE_POSITIVES = "n_false_positives"
    N_FALSE_NEGATIVES = "n_false_negatives"
    IOU = "IoU"
    JACCARD = "Jaccard"
    PIXEL_IDENTITY = "pixel_identity"
    LOCALIZATION_ERROR = "localization_error"


METRICS = (
    "n_true_labels",
    "n_pred_labels",
    "n_true_positives",
    "n_false_positives",
    "n_false_negatives",
    "IoU",
    "Jaccard",
    "pixel_identity",
    "localization_error",
)


def _IoU(ref: npt.NDArray, pred: npt.NDArray) -> float:
    """Calculate the IoU between two binary masks."""
    intersection = np.sum(np.logical_and(ref, pred))
    union = np.sum(np.logical_or(ref, pred))
    iou = 0.0 if union == 0 else intersection / union
    return iou


def find_matches(
    ref: LabeledSegmentation,
    pred: LabeledSegmentation,
    *,
    strict: bool = False,
    iou_threshold: float = 0.5,
) -> Dict:
    """Perform matching between the reference and the predicted image.

    Parameters
    ----------
    ref :
        The reference (ground truth) segmentation.
    pred :
        The predicted segmentation.

    Return
    ------
    matches : dict
        A dictionary of matches between the two images.

    """

    # return a default dictionary of no matches
    matches = {
        "true_matches": [],
        "true_matches_IoU": [],
        "in_ref_only": set(ref.labels),
        "in_pred_only": set(pred.labels),
    }

    # make an infinite cost matrix, so that we only consider matches where
    # there is some overlap in the masks
    cost_matrix = np.full((len(ref.labels), len(pred.labels)), np.inf)

    for r_id, ref_label in enumerate(ref.labels):
        mask = ref.labeled == ref_label
        _matches = [m for m in np.unique(pred.labeled[mask]) if m > 0]
        for pred_label in _matches:
            p_id = pred.labels.index(pred_label)
            reward = _IoU(mask, pred.labeled == pred_label)
            if reward < iou_threshold and strict:
                continue
            cost_matrix[r_id, p_id] = 1.0 - reward

    # if it's strict, make sure every element is above the threshold
    if strict:
        assert np.all(cost_matrix >= iou_threshold)

    try:
        sol_row, sol_col = linear_sum_assignment(cost_matrix)
    except ValueError:
        return matches

    # now that we've solved the LAP, find the matches that have been made
    used_ref = [ref.labels[row] for row in sol_row]
    used_pred = [pred.labels[col] for col in sol_col]
    assert len(used_ref) == len(used_pred)
    true_matches = list(zip(used_ref, used_pred))

    # find the labels that haven't been used
    in_ref_only = set(ref.labels).difference(used_ref)
    in_pred_only = set(pred.labels).difference(used_pred)

    # return a dictionary of found matches
    matches = {
        "true_matches": true_matches,
        "true_matches_IoU": 1.0 - cost_matrix[sol_row, sol_col],
        "in_ref_only": in_ref_only,
        "in_pred_only": in_pred_only,
    }

    return matches


class MetricResults(object):
    def __init__(self, metrics):
        assert isinstance(metrics, SegmentationMetrics)
        self._images = 1
        self._metrics = metrics

        # list of metrics that are aggregated
        self._agg = (
            "n_true_labels",
            "n_pred_labels",
            "n_true_positives",
            "n_false_positives",
            "n_false_negatives",
            "per_object_IoU",
            "per_object_localization_error",
            "per_image_pixel_identity",
        )

    def __getattr__(self, key):
        return getattr(self._metrics, key)

    @property
    def n_images(self) -> int:
        if any([getattr(self, m) is None for m in self._agg]):
            return 0
        else:
            return self._images

    def __add__(self, result: MetricResults) -> MetricResults:
        assert isinstance(result, MetricResults)
        for m in self._agg:
            setattr(self, m, getattr(result, m) + getattr(self, m))
        self._images += 1
        return self

    def __repr__(self) -> str:
        title = f" Segmentation Metrics (n={self.n_images})\n"
        hbar = "=" * len(title) + "\n"
        r = hbar + title + hbar
        if self.strict:
            r += f"Strict: {self.strict} (IoU > {self.iou_threshold})\n"
        for m in METRICS:
            mval = getattr(self, m)
            if isinstance(mval, float):
                r += f"{m}: {mval:.3f}\n"
            else:
                r += f"{m}: {mval}\n"
        return r

    @property
    def localization_error(self) -> float:
        return np.mean(self.per_object_localization_error)

    @property
    def IoU(self) -> float:
        return np.mean(self.per_object_IoU)

    @property
    def Jaccard(self) -> float:
        """Jaccard metric"""
        tp = self.n_true_positives
        fn = self.n_false_negatives
        fp = self.n_false_positives
        return tp / (tp + fn + fp)

    @property
    def pixel_identity(self) -> float:
        return np.mean(self.per_image_pixel_identity)

    @staticmethod
    def merge(results) -> MetricResults:
        """merge n results together and return a single object"""
        assert isinstance(results, list)
        merged = results.pop(0)
        for result in results:
            assert isinstance(result, MetricResults)
            assert result.n_images == 1
            merged = merged + result
        return merged


class SegmentationMetrics:
    """A class for calculating various segmentation metrics to assess the
    accuracy of a trained model.

    Parameters
    ----------
    reference : array
        An array containing labeled objects from the ground truth.
    predicted : array
        An array containing labeled objects from the segmentation algorithm.
    strict : bool
        Whether to disregard matches with a low IoU score.
    iou_threshold : float
      Threshold IoU for strict matching.

    Properties
    ----------
    Jaccard : float
        The Jaccard index calculated according to the notes below.
    IoU : float
        The Intersection over Union metric.
    localisation_precision : float
        The localisation precision.
    true_positives : int
        Number of TP predictions.
    false_positives : int
        Number of FP predictions.
    false_negatives : int
        Number of FN predicitons.


    Notes
    -----
    The Jaccard metric is calculated accordingly:

        FP = number of objects in predicted but not in reference
        TP = number of objects in both
        TN = background correctly segmented (not used)
        FN = number of objects in true but not in predicted

        J = TP / (TP+FP+FN)

    The IoU is calculated as the intersection of the binary segmentation
    divided by the union.

    TODO(arl): need to address undersegmentation detection

    """

    def __init__(
        self, reference: LabeledSegmentation, predicted: LabeledSegmentation, **kwargs
    ):
        assert isinstance(predicted, LabeledSegmentation)
        assert isinstance(reference, LabeledSegmentation)

        self._reference = reference
        self._predicted = predicted
        self._strict = kwargs.get("strict", False)
        self._iou_threshold = kwargs.get("iou_threshold", 0.5)

        if self.iou_threshold < 0.0 or self.iou_threshold > 1.0:
            raise ValueError(
                f"IoU Threshold shoud be in (0, 1), found: {self.iou_threshold:.2f}"
            )
        assert isinstance(self.strict, bool)

        # find the matches
        self._matches = find_matches(
            self._reference,
            self._predicted,
            strict=self.strict,
            iou_threshold=self.iou_threshold,
        )

        # # if we're in strict mode, prune the matches
        # if self.strict:
        #     iou = self.per_object_IoU
        #     tp = [
        #         self.true_positives[i]
        #         for i, ov in enumerate(iou)
        #         if ov > self.iou_threshold
        #     ]
        #     fp = list(set(self.true_positives).difference(tp))

        #     self._matches["true_matches"] = tp
        #     self._matches["in_pred_only"] += [m[1] for m in fp]

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @property
    def results(self):
        return MetricResults(self)

    @property
    def image_overlay(self):
        # n_labels = max([self._predicted.n_labels, self._reference.n_labels])
        # scale = int(255 / n_labels)
        return (
            np.stack(
                [self._predicted.image, self._reference.image, self._predicted.image],
                axis=-1,
            )
            * 127
        )

    @property
    def n_true_labels(self):
        return self._reference.n_labels

    @property
    def n_pred_labels(self):
        return self._predicted.n_labels

    @property
    def true_positives(self):
        """only one match between reference and predicted"""
        return self._matches["true_matches"]

    @property
    def false_negatives(self):
        """no match in predicted for reference object"""
        return self._matches["in_ref_only"]

    @property
    def false_positives(self):
        """combination of non unique matches and unmatched objects"""
        return self._matches["in_pred_only"]

    @property
    def n_true_positives(self):
        return len(self.true_positives)

    @property
    def n_false_negatives(self):
        return len(self.false_negatives)

    @property
    def n_false_positives(self):
        return len(self.false_positives)

    @property
    def per_object_IoU(self):
        """Intersection over Union (IoU) metric"""
        iou = []
        for m in self.true_positives:
            mask_ref = self._reference.labeled == m[0]
            mask_pred = self._predicted.labeled == m[1]

            iou.append(_IoU(mask_ref, mask_pred))
        return iou

    @property
    def per_image_pixel_identity(self):
        n_tot = np.prod(self._reference.image.shape)
        return [np.sum(self._reference.image == self._predicted.image) / n_tot]

    @property
    def per_object_localization_error(self):
        """localization error"""
        ref_centroids = self._reference.centroids
        tgt_centroids = self._predicted.centroids
        positional_error = []
        for m in self.true_positives:
            true_centroid = np.array(ref_centroids[m[0] - 1])
            pred_centroid = np.array(tgt_centroids[m[1] - 1])
            err = np.sum((true_centroid - pred_centroid) ** 2)
            positional_error.append(err)
        return positional_error

    def plot(self):
        render.plot_metrics(self)

    def to_napari(self):
        return render.render_metrics_napari(self)


class LabeledSegmentation:
    """LabeledSegmentation

    A helper class to enable simple calculation of accuracy statistics for
    image segmentation output.

    """

    def __init__(self, image: npt.NDArray):
        self.image = image
        self.labeled, self.n_labels = label(image.astype(bool))

    @property
    def shape(self):
        return self.image.shape

    @property
    def bboxes(self):
        return [find_objects(self.labeled == label)[0] for label in self.labels]

    @property
    def labels(self):
        return range(1, self.n_labels + 1)

    @property
    def centroids(self):
        return [center_of_mass(self.labeled == label) for label in self.labels]

    @property
    def areas(self):
        return [np.sum(self.labeled == label) for label in self.labels]


def calculate(reference, predicted, **kwargs):
    """Take a predicted image and compare with the reference image.

    Compute various metrics.
    """

    ref = LabeledSegmentation(reference)
    tgt = LabeledSegmentation(predicted)

    # make sure they are the same size
    assert ref.shape == tgt.shape

    return SegmentationMetrics(ref, tgt, **kwargs)


def batch(files, **kwargs):
    """batch process a list of files"""
    metrix = []
    for f_ref, f_pred in files:
        print(f_pred)
        true = imread(f_ref)
        pred = imread(f_pred)
        result = calculate(true, pred, **kwargs).results
        metrix.append(result)
    return MetricResults.merge(metrix)


if __name__ == "__main__":
    pass
