# UNet segmentation metrics

*WORK IN PROGRESS*

Simple Python 3 tools to assess the performance of UNet segmentation networks
(or any other segmentation method) by comparing the prediction to a ground truth
 image.

Use it to calculate:
+ Jaccard metric for object detection
+ Intersection over Union (IoU) for object segmentation accuracy
+ Localization (positional) error for estimating MOTP during tracking
+ Pixel identity

TODO:
+ [x] Add strict matching with IoU threshold  
+ [ ] Add confusion matrix for multi-label/classification type tasks


### Single image usage

```python
import umetrix
from skimage.io import imread

y_true = imread('true.tif')
y_pred = imread('pred.tif')


# can now make the calculation strict, by only considering objects that have
# an IoU above a theshold as being true positives
result = umetrix.calculate(
    y_true,
    y_pred,
    strict=True,
    iou_threshold=0.5
)

print(result.results)
```

returns:

```
============================
 Segmentation Metrics (n=1)
============================
Strict: True (IoU > 0.5)
n_true_labels: 354
n_pred_labels: 362
n_true_positives: 345
n_false_positives: 10
n_false_negatives: 0
IoU: 0.999
Jaccard: 0.972
pixel_identity: 0.998
localization_error: 0.010
```


### Batch processing

```python
import umetrix

# provide a list of file pairs ('true', 'prediction')
files = [
    ('true0.tif', 'pred0.tif'),
    ('true1.tif', 'pred1.tif'),
    ('true2.tif', 'pred2.tif')
]

batch_result = umetrix.batch(files)
```

Returns aggregate statistics over the batch. Jaccard index is calculated over
all found objects, while other metrics are the average IoU etc.


### Installation

1. First clone the repo:
```sh
$ git clone https://github.com/quantumjot/unet_segmentation_metrics.git
```

2. (Optional, but advised) Create a conda environment:
```sh
$ conda create -n umetrix python=3.9
$ conda activate umetrix
```

3. Install the package
```sh
$ cd unet_segmentation_metrics
$ pip install .
```
