
__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_metrics(seg_metrics):

    pred = seg_metrics._predicted
    ref = seg_metrics._reference

    iou = [None] * len(ref.labels)
    IoU = seg_metrics.per_object_IoU
    for i, tp in enumerate(seg_metrics.true_positives):
        iou[tp[0]-1] = '{:.2f}'.format(IoU[i])

    fig, ax = plt.subplots(1, figsize=(16,12))
    # plt.imshow(J_image)
    ax.imshow(seg_metrics.image_overlay)


    for i, (sy, sx) in enumerate(ref.bboxes):
        r = patches.Rectangle((sx.start,sy.start), sx.stop-sx.start,sy.stop-sy.start, edgecolor='g', facecolor='None')
        ax.add_patch(r)
        ax.text(sx.start, sy.start, '{}, IoU: {}'.format(i, iou[i]), fontsize=6, color='w')
    for i, (sy, sx) in enumerate(pred.bboxes):
        r = patches.Rectangle((sx.start,sy.start), sx.stop-sx.start,sy.stop-sy.start, edgecolor='m', facecolor='None')
        ax.add_patch(r)
        # ax.text(sx.start, sy.start, '{}'.format(i), fontsize=6, color='w')

    bboxes = pred.bboxes
    for fp in seg_metrics.false_positives:
        sy, sx = bboxes[fp-1]
        w, h = sx.stop-sx.start,sy.stop-sy.start
        r = patches.Rectangle((sx.start,sy.start), w, h, edgecolor='r', facecolor=(1.,0.,0.,0.0), linewidth=2)
        ax.add_patch(r)

    bboxes = ref.bboxes
    for fn in seg_metrics.false_negatives:
        sy, sx = bboxes[fn-1]
        w, h = sx.stop-sx.start,sy.stop-sy.start
        r = patches.Rectangle((sx.start,sy.start), w, h, edgecolor='c', facecolor=(0.,1.,1.,0.0), linewidth=2)
        ax.add_patch(r)
    plt.axis('off')
    plt.show()
