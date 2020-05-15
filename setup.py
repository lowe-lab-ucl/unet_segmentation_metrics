#!/usr/bin/env python
from setuptools import setup

setup(name='unet_segmentation_metrics',
      version='0.1',
      description='Segmentation performance metrics',
      author='Alan R. Lowe',
      author_email='a.lowe@ucl.ac.uk',
      url='https://github.com/quantumjot/unet_segmentation_metrics',
      packages=['umetrics'],
      install_requires=['numpy','scipy','matplotlib','scikit-image'],
      python_requires='>=3.6'
     )
