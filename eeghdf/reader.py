# -*- coding: utf-8 -*-
"""
functions to help with reading eeghdf files
versions 1...
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

def record_edf_annotations_to_lists(raw_edf_annotations):
    """
    usage: 

    >>> rec = hdf['record-0']
    >>> texts, times_100ns = record_edf_annotations_to_lists(rec['edf_annotations'])
    """
    byte_texts = raw_edf_annotations['texts'] # still byte encoded
    antexts = [s.decode('utf-8') for s in byte_texts[:]]

    starts100ns_arr = raw_edf_annotations['starts_100ns'][:]
    starts100ns = [xx for xx in starts100ns_arr]
    return antexts, starts100ns

def record_edf_annotations_to_sec_items(raw_edf_annotations):
    """
    rec = hdf['record-0']
    annotation_items = record_edf_annotations_to_sec_items(rec['edf_annotations'])
    # returns (text, <start time (sec)>) pairs
    """
    byte_texts = raw_edf_annotations['texts'] # still byte encoded
    antexts = [s.decode('utf-8') for s in byte_texts[:]]

    starts100ns_arr = raw_edf_annotations['starts_100ns'][:]
    starts_sec_arr = starts100ns_arr/10000000  #  (10**7) * 100ns = 1 second 
    items = zip(antexts, starts_sec_arr)
    return items
