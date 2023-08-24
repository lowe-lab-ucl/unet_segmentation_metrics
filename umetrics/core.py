
__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import os

import numpy as np

from skimage.io import imread
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from scipy.ndimage import find_objects

from . import render


METRICS = ('n_true_labels',
           'n_pred_labels',
           'n_true_positives',
           'n_false_positives',
           'n_false_negatives',
           'IoU',
           'Jaccard',
           'pixel_identity',
           'localization_error')



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
    """ Find matches between the reference image and the predicted image.

    Args:
        ref
        pred
    """

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
                raise Exception('Something doesn\'t make sense...')
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



class MetricResults(object):
    def __init__(self, metrics):
        assert(isinstance(metrics, SegmentationMetrics))
        self._images = 1
        self._metrics = metrics

        # list of metrics that are aggregated
        self._agg = ('n_true_labels',
                     'n_pred_labels',
                     'n_true_positives',
                     'n_false_positives',
                     'n_false_negatives',
                     'per_object_IoU',
                     'per_object_localization_error',
                     'per_image_pixel_identity')

    def __getattr__(self, key):
        return getattr(self._metrics, key)

    @property
    def n_images(self):
        if any([getattr(self, m) is None for m in self._agg]):
            return 0
        else:
            return self._images

    def __add__(self, result):
        assert(isinstance(result, MetricResults))
        for m in self._agg:
            setattr(self, m, getattr(result, m)+getattr(self, m))
        self._images+=1
        return self

    def __repr__(self):
        title = f' Segmentation Metrics (n={self.n_images})\n'
        hbar = '='*len(title)+'\n'
        r = hbar + title + hbar
        if self.strict:
            r += f'Strict: {self.strict} (IoU > {self.iou_threshold})\n'
        for m in METRICS:
            mval = getattr(self,m)
            if isinstance(mval, float):
                r+= f'{m}: {mval:.3f}\n'
            else:
                r+= f'{m}: {mval}\n'
        return r


    @property
    def localization_error(self):
        return np.mean(self.per_object_localization_error)

    @property
    def IoU(self):
        return np.mean(self.per_object_IoU)

    @property
    def Jaccard(self):
        """ Jaccard metric """
        tp = self.n_true_positives
        fn = self.n_false_negatives
        fp = self.n_false_positives
        return tp / (tp+fn+fp)

    @property
    def pixel_identity(self):
        return np.mean(self.per_image_pixel_identity)


    @staticmethod
    def merge(results):
        """ merge n results together and return a single object """
        assert(isinstance(results, list))
        merged = results.pop(0)
        for result in results:
            assert(isinstance(result, MetricResults))
            assert(result.n_images == 1)
            merged = merged + result
        return merged



class SegmentationMetrics:
    """ SegmentationMetrics

        A class for calculating various segmentation metrics to assess the
        accuracy of a trained model.

        Args:
            reference - a numpy array (wxh) containing labeled objects from the
                ground truth
            predicted - a numpy array (wxh) containing labeled objects from the
                segmentation algorithm

            strict - (bool) whether to disregard matches with a low IoU score
            iou_threshold - (float) threshold for strict matching

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
    def __init__(self,
                 reference,
                 predicted,
                 **kwargs):

        assert(isinstance(predicted, LabeledSegmentation))
        assert(isinstance(reference, LabeledSegmentation))


        self._reference = reference
        self._predicted = predicted
        self._strict = kwargs.get('strict', False)
        self._iou_threshold = kwargs.get('iou_threshold', 0.5)

        assert(self.iou_threshold>=0. and self.iou_threshold<=1.)
        assert(isinstance(self.strict, bool))

        # find the matches
        self._matches = find_matches(self._reference, self._predicted)

        # if we're in strict mode, prune the matches
        if self.strict:
            iou = self.per_object_IoU
            tp = [self.true_positives[i] for i, ov in enumerate(iou) if ov > self.iou_threshold]
            fp = list(set(self.true_positives).difference(tp))

            self._matches['true_matches'] = tp
            self._matches['in_pred_only'] += [m[1] for m in fp]

    @property
    def strict(self):
        return self._strict

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @property
    def results(self):
        return MetricResults(self)

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
    def n_true_positives(self): return len(self.true_positives)
    @property
    def n_false_negatives(self): return len(self.false_negatives)
    @property
    def n_false_positives(self): return len(self.false_positives)

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
    def per_image_pixel_identity(self):
        n_tot = np.prod(self._reference.image.shape)
        return [np.sum(self._reference.image == self._predicted.image) / n_tot]

    @property
    def per_object_localization_error(self):
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

    def plot(self):
        render.plot_metrics(self)

    def to_napari(self):
        return render.render_metrics_napari(self)




class LabeledSegmentation(object):
    """ LabeledSegmentation

    A helper class to enable simple calculation of accuracy statistics for
    image segmentation output.

    """
    def __init__(self, image):
        assert(isinstance(image, np.ndarray))
        self.image = image

        # label it
        self.labeled, self.n_labels = label(image.astype(bool))

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






def calculate(reference, predicted, **kwargs):
    """ Take a predicted image and compare with the reference image.

    Compute various metrics.
    """

    ref = LabeledSegmentation(reference)
    tgt = LabeledSegmentation(predicted)

    # make sure they are the same size
    assert(ref.shape == tgt.shape)

    return SegmentationMetrics(ref, tgt, **kwargs)


def batch(files, **kwargs):
    """ batch process a list of files """
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
