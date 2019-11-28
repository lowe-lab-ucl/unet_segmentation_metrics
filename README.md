# UNet segmentation metrics

*WORK IN PROGRESS*

Simple tools to assess the performance of UNet segmentation networks by comparing the ground truth image to the prediction.

Use it to calculate:
+ Jaccard metric for object detection
+ Intersection over Union (IoU) for object segmentation accuracy
+ Localization (positional) error
+ Pixel identity

### Usage

```python
from segmetrics import metrics
from skimage.io import imread

true = imread('true.tif')
pred = imread('pred.tif')

m = metrics.calculate(true, pred)

print(m)

```

returns:

```
UNet Segmentation Metrics:
==========================
True objects:      336
Predicted objects: 320
True positives:    315
False positives:    13
False negatives:    17
Jaccard metric:      0.91
Mean IoU metric:     0.83
Pixel identity:      0.97
```
