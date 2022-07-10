# -*- coding: utf-8 -*-
"""eeghdf is a module for reading a writing EEG data into the hdf5 format
Features include:
  - efficient storage of EEG data
  - reading portions of EEG data without reading the whole file using a numpy-array like interface
  - streaming from S3 buckets using the ROS3 driver
"""
from __future__ import absolute_import

from .writer import EEGHDFWriter
from .reader import *

__version__ = "0.2.2"
