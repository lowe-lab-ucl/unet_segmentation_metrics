
__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import os

import numpy as np

from skimage.io import imread
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from scipy.ndimage import find_objects

from . import render



class SegmentationMetrics(object):
    """ SegmentationMetrics

        A class for calculating various segmentation metrics to assess the
        accuracy of a trained model.

        Args:
            reference - a numpy array (wxh) containing labeled objects from the
                ground truth
            target - a numpy array (wxh) containing labeled objects from the
                segmentation algorithm

        Members:
            find_matches: find overlapping regions of the two images

        Properties:
            Jaccard: the Jaccard index calculated according to the notes below
            IoU: the Intersection over Union metric
            localisation_precision:

            true_positives:
            false_positives:
            false_negatives:


        Notes:
            The Jaccard metric is calculated accordingly:

                FP = number of objects in target but not in reference
                TP = number of objects in both
                TN = background correctly segmented (not used)
                FN = number of objects in reference but not in target

                J = TP / (TP+FP+FN)

            The IoU is calculated as the intersection of the binary segmentation
            divided by the union.

    """
    def __init__(self, reference, target):
        assert(isinstance(target, LabeledSegmentation))
        assert(isinstance(reference, LabeledSegmentation))
        self._reference = reference
        self._target = target

        self._matches = self.find_matches()

    def find_matches(self):
        """ find potential matches between objects in the reference and
        target images. These need to have at least 1 pixel of overlap.
        """
        matches = []
        for label in self._reference.labels:
            mask = self._reference.labeled == label
            match = [m for m in np.unique(self._target.labeled[mask]) if m>0]
            matches.append((label, match))
        return matches

    @property
    def image_overlay(self):
        return np.stack([self._target.image,
                         self._reference.image,
                         self._target.image],axis=-1)*127

    @property
    def true_matches(self):
        """ present in both reference and target """
        return [m for m in self._matches if len(m[1]) == 1]

    @property
    def non_unique_matches(self):
        """ oversegmentation """
        multi = []
        for m in self._matches:
            if len(m[1]) > 1:
                multi += m[1]
        return multi

    @property
    def unmatched(self):
        """ objects in target, not in reference, i.e. hallucinations  """
        tgt_matches = []
        for m in self._matches:
            tgt_matches += m[1]
        return list(set(self._target.labels).difference(set(tgt_matches)))

    @property
    def true_positives(self):
        """ only one match between reference and target """
        return [m[0] for m in self.true_matches]

    @property
    def false_negatives(self):
        """ no match in target for reference object """
        return [m[0] for m in self._matches if len(m[1]) < 1]

    @property
    def false_positives(self):
        """ combination of non unique matches and unmatched objects """
        return self.non_unique_matches + self.unmatched

    @property
    def Jaccard(self):
        """ Jaccard metric """
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        fp = len(self.false_positives)
        return tp / (tp+fn+fp)

    @property
    def IoU(self):
        """ Intersection over Union (IoU) metric """
        iou = []
        for m in self.true_matches:
            mask_ref = self._reference.labeled == m[0]
            mask_tgt = self._target.labeled == m[1]

            intersection = np.logical_and(mask_ref, mask_tgt)
            union = np.logical_or(mask_ref, mask_tgt)

            iou.append(np.sum(intersection)/np.sum(union))
        return iou

    @property
    def pixel_identity(self):
        n_total = np.prod(self._reference.image.shape)
        return np.sum(self._reference.image == self._target.image) / n_total

    @property
    def localization_error(self):
        """ localization error """
        raise NotImplementedError
        ref_centroids = self._reference.centroids
        tgt_centroids = self._target.centroids
        positional_error = []
        for m in self.true_matches:
            true_centroid = np.array(ref_centroids[m[0]-1])
            pred_centroid = np.array(tgt_centroids[m[1][0]-1])
            err = np.sum((true_centroid-pred_centroid)**2)
            positional_error.append(err)
        return positional_error


    def __repr__(self):
        repr = "\nUNet Segmentation Metrics: \n"
        repr += "================================ \n"
        repr += "True objects: \t\t{:>5}\n".format(len(self._reference.labels))
        repr += "Predicted objects: \t{:>5}\n".format(len(self._target.labels))
        repr += "True positives: \t{:>5}\n".format(len(self.true_positives))
        repr += "False positives: \t{:>5}\n".format(len(self.false_positives))
        repr += "False negatives: \t{:>5}\n".format(len(self.false_negatives))
        repr += "Jaccard metric: \t{:>8.2f}\n".format(self.Jaccard)
        repr += "Mean IoU metric: \t{:>8.2f}\n".format(np.mean(self.IoU))
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






def calculate(reference, target):
    """ Take a target image and compare with the reference image.

    Compute various metrics.
    """

    ref = LabeledSegmentation(reference)
    tgt = LabeledSegmentation(target)

    # make sure they are the same size
    assert(ref.shape == tgt.shape)

    return SegmentationMetrics(ref, tgt)








if __name__ == "__main__":

    p = '/Users/arl/Dropbox/Data/TestingData/UNet2D_testing_Scribble/set12'
    ref = imread(os.path.join(p, 'labels_compressed', 'l_7.tif'))
    tgt = imread(os.path.join(p, 'segmented_2019-11-27', 's_7.tif'))

    m = calculate(ref, tgt)
    print(m)
    m.plot()
