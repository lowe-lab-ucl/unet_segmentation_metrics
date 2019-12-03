
__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import os

import numpy as np

from skimage.io import imread
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from scipy.ndimage import find_objects

from . import render


def _find_matches(ref, pred):
    """ find potential matches between objects in the reference and
    predicted images. These need to have at least 1 pixel of overlap.
    """
    matches = {}
    for label in ref.labels:
        mask = ref.labeled == label
        matches[label] = [m for m in np.unique(pred.labeled[mask]) if m>0]
    return matches


def find_matches(ref, pred):
    # do forward and reverse matching
    matches_rp = _find_matches(ref, pred)
    matches_pr = _find_matches(pred, ref)

    true_matches = []
    in_ref_only = []
    in_pred_only = []

    for m0, match in matches_rp.items():
        # no matches
        if len(match) < 1:
            in_ref_only.append(m0)
        # if there is only one match, check that there is only one reverse match
        elif len(match) == 1:
            if match[0] not in matches_pr:
                print('oof', match[0], matches_pr)
            elif len(matches_pr[match[0]]) == 1:
                # one to one
                true_matches.append((m0, match[0]))
            elif len(matches_pr[match[0]]) > 1:
                # two (or more) objects in the prediction match one in the ref
                in_pred_only += match
        elif len(match) > 1:
            in_pred_only += match

    # sanity check that all are accounted for
    ref_found_labels = set(in_ref_only + [m[0] for m in true_matches])
    pred_found_labels = set(in_pred_only + [m[1] for m in true_matches])

    assert( len(ref_found_labels.difference(set(ref.labels))) == 0 )
    assert( len(pred_found_labels.difference(set(pred.labels))) == 0 )

    # return a dictionary of found matches
    matches = {'true_matches': true_matches,
               'in_ref_only': list(set(in_ref_only)),
               'in_pred_only': list(set(in_pred_only))}
    return matches


class SegmentationMetrics(object):
    """ SegmentationMetrics

        A class for calculating various segmentation metrics to assess the
        accuracy of a trained model.

        Args:
            reference - a numpy array (wxh) containing labeled objects from the
                ground truth
            predicted - a numpy array (wxh) containing labeled objects from the
                segmentation algorithm

        Properties:
            Jaccard: the Jaccard index calculated according to the notes below
            IoU: the Intersection over Union metric
            localisation_precision:

            true_positives:
            false_positives:
            false_negatives:


        Notes:
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
    def __init__(self, reference, predicted):
        assert(isinstance(predicted, LabeledSegmentation))
        assert(isinstance(reference, LabeledSegmentation))
        self._reference = reference
        self._predicted = predicted

        # find the matches
        self._matches = find_matches(self._reference, self._predicted)


    @property
    def metrics(self):
        return [self.n_true_labels,
                self.n_pred_labels,
                len(self.true_positives),
                len(self.false_positives),
                len(self.false_negatives),
                self.IoU,
                self.Jaccard,
                self.pixel_identity]

    @property
    def image_overlay(self):
        n_labels = max([self._predicted.n_labels, self._reference.n_labels])
        scale = int(255/n_labels)
        return np.stack([self._predicted.image,
                         self._reference.image,
                         self._predicted.image],axis=-1)*127

    @property
    def n_true_labels(self):
        return self._reference.n_labels

    @property
    def n_pred_labels(self):
        return self._predicted.n_labels

    @property
    def true_positives(self):
        """ only one match between reference and predicted """
        return self._matches['true_matches']

    @property
    def false_negatives(self):
        """ no match in predicted for reference object """
        return self._matches['in_ref_only']

    @property
    def false_positives(self):
        """ combination of non unique matches and unmatched objects """
        return self._matches['in_pred_only']

    @property
    def Jaccard(self):
        """ Jaccard metric """
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        fp = len(self.false_positives)
        return tp / (tp+fn+fp)

    @property
    def per_object_IoU(self):
        """ Intersection over Union (IoU) metric """
        iou = []
        for m in self.true_positives:
            mask_ref = self._reference.labeled == m[0]
            mask_pred = self._predicted.labeled == m[1]

            intersection = np.logical_and(mask_ref, mask_pred)
            union = np.logical_or(mask_ref, mask_pred)

            iou.append(np.sum(intersection)/np.sum(union))
        return iou

    @property
    def IoU(self):
        return np.mean(self.per_object_IoU)

    @property
    def pixel_identity(self):
        n_total = np.prod(self._reference.image.shape)
        return np.sum(self._reference.image == self._predicted.image) / n_total

    @property
    def localization_error(self):
        """ localization error """
        ref_centroids = self._reference.centroids
        tgt_centroids = self._predicted.centroids
        positional_error = []
        for m in self.true_positives:
            true_centroid = np.array(ref_centroids[m[0]-1])
            pred_centroid = np.array(tgt_centroids[m[1]-1])
            err = np.sum((true_centroid-pred_centroid)**2)
            positional_error.append(err)
        return positional_error

    def __repr__(self):
        repr = "\nUNet Segmentation Metrics: \n"
        repr += "================================ \n"
        repr += "True objects: \t\t{:>5}\n".format(self._reference.n_labels)
        repr += "Predicted objects: \t{:>5}\n".format(self._predicted.n_labels)
        repr += "True positives: \t{:>5}\n".format(len(self.true_positives))
        repr += "False positives: \t{:>5}\n".format(len(self.false_positives))
        repr += "False negatives: \t{:>5}\n".format(len(self.false_negatives))
        repr += "Jaccard metric: \t{:>8.2f}\n".format(self.Jaccard)
        repr += "Mean IoU metric: \t{:>8.2f}\n".format(self.IoU)
        repr += "Pixel identity: \t{:>8.2f}\n".format(self.pixel_identity)
        # repr += "Mean localization error: \t{:.2f}\n".format(np.mean(self.localization_error))
        return repr

    def plot(self):
        render.plot_metrics(self)




class LabeledSegmentation(object):
    """ LabeledSegmentation

    A helper class to enable simple calculation of accuracy statistics for
    image segmentation output.

    """
    def __init__(self, image):
        assert(isinstance(image, np.ndarray))
        self.image = image

        # label it
        self.labeled, self.n_labels = label(image.astype(np.bool))

    @property
    def shape(self):
        return self.image.shape

    @property
    def bboxes(self):
        return [find_objects(self.labeled==label)[0] for label in self.labels]

    @property
    def labels(self):
        return range(1, self.n_labels+1)

    @property
    def centroids(self):
        return [center_of_mass(self.labeled==label) for label in self.labels]

    @property
    def areas(self):
        return [np.sum(self.labeled==label) for label in self.labels]






def calculate(reference, predicted):
    """ Take a predicted image and compare with the reference image.

    Compute various metrics.
    """

    ref = LabeledSegmentation(reference)
    tgt = LabeledSegmentation(predicted)

    # make sure they are the same size
    assert(ref.shape == tgt.shape)

    return SegmentationMetrics(ref, tgt)


def batch(files):
    """ batch process a list of files """
    metrix = []
    for f_ref, r_pred in files:
        true = imread(f_ref)
        pred = imread(f_pred)
        metrix.append(calculate(true, pred))
    return metrix








if __name__ == "__main__":
    pass
