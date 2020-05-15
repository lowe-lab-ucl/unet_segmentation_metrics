# UNet segmentation metrics

*WORK IN PROGRESS*

Simple Python 3 tools to assess the performance of UNet segmentation networks by comparing the ground truth image to the prediction.

Use it to calculate:
+ Jaccard metric for object detection
+ Intersection over Union (IoU) for object segmentation accuracy
+ Localization (positional) error for estimating MOTP during tracking
+ Pixel identity


### Single image usage

```python
import umetrics
from skimage.io import imread

true = imread('true.tif')
pred = imread('pred.tif')

result = umetrics.calculate(true, pred)

print(result)
```

returns:

```
============================
 Segmentation Metrics (n=1)
============================
n_true_labels: 110
n_pred_labels: 103
n_true_positives: 97
n_false_positives: 6
n_false_negatives: 8
IoU: 0.838
Jaccard: 0.874
pixel_identity: 0.991
localization_error: 2.635
```


### Batch processing

```python
import umetrics

# provide a list of file pairs ('true', 'prediction')
files = [('true0.tif', 'pred0.tif'),
         ('true1.tif', 'pred1.tif'),
         ('true2.tif', 'pred2.tif')]

batch_result = umetrics.batch(files)

print(batch_result)
```

Returns aggregate statistics over the batch. Jaccard index is calculated over all found objects, while other metrics are the average IoU etc.


### Installation

1. First clone the repo:
```sh
$ git clone https://github.com/quantumjot/unet_segmentation_metrics.git
```

2. (Optional, but advised) Create a conda environment:
```sh
$ conda create -n umetrics
$ conda activate umetrics
```

3. Install the package
```sh
$ cd unet_segmentation_metrics
$ pip install .
```
