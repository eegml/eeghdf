# -*- coding: utf-8 -*-
"""
functions to help with reading eeghdf files
versions 1...
"""
# python 2/3 compatibility - write as if in python 3.5
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
from past.utils import old_div

import h5py
import pandas as pd # maybe shouldn't require pandas in basic file so look to eliminate


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


# so given a hdf 
# signals(integers) -> optional-uV-conversion -> optional montage conversion (???scope of this project)
# 

## let's try a first draft to get a feel for things


class Eeghdf:
    __version__ = 1
    def __init__(self, fn, mode='r'):
        """
        version 1: assumes only one record-0 waveform
        but may allow for record_list in future

        h5py.File mode options
        r readonly
        r+ read/write, file must exist
        w create file, truncate if exists
        w- or x create file, fail if exists
        a read/write if exists, create otherwise
        """
        self.file_name = fn
        self.hdf = h5py.File(fn, mode=mode) # readonly by default



        # waveform record info
        self.rec0 = self.hdf['record-0']
        rec0 = self.rec0
        self.age_years = rec0.attrs['patient_age_days'] / 365 # age of patient at time of record

        self.rawsignals = rec0['signals']
        labels_bytes = rec0['signal_labels']
        self.electrode_labels = [str(s,'ascii') for s in labels_bytes]


        # read annotations and format them for easy use
        annot = rec0['edf_annotations'] # but these are in a funny 3 array format
        antext = [s.decode('utf-8') for s in annot['texts'][:]]
        self._annotation_text = antext
        starts100ns = [xx for xx in annot['starts_100ns'][:]]
        self._annotation_start100s = starts100ns
        start_time_sec = [xx/10**7 for xx in starts100ns] # 10**7 * 100ns = 1sec
        df = pd.DataFrame(data=antext,columns=['text'])
        df['starts_sec'] = start_time_sec
        df['starts_100ns'] = starts100ns
        self.edf_annotations_df = df

        self._physical_dimensions = None

        # what about units and conversion factors
        self.start_isodatetime = rec0.attrs['start_isodatetime'] # = start_isodatetime
        self.end_isodatetime = rec0.attrs['end_isodatetime'] # = end_isodatetime

        self.number_channels = rec0.attrs['number_channels'] # = number_channels
        self.number_samples_per_channel = rec0.attrs['number_samples_per_channel'] # = num_samples_per_channel
        self.sample_frequency = rec0.attrs['sample_frequency'] # = sample_frequency


        # record['signal_digital_maxs'] = signal_digital_maxs
        # rec0.attrs['patient_age_days'] = patient_age_days
        # rec0.attrs['technician'] = technician
        # patient
        self.patient = dict(self.hdf['patient'].attrs)

    def annotations_contain(self, pat, case=False):
        df = self.edf_annotations_df
        return df[df.text.str.contains(pat,case=case)]

    @property
    def physical_dimensions(self):
        if not self._physical_dimensions:
            self._physical_dimensions = [s.decode('utf-8') for s in self.rec0['physical_dimensions'][:]]
        return self._physical_dimensions

    @property
    def signal_physical_mins(self):
        return self.hdf['record-0']['signal_physical_mins'][:]
    
    #    self.# record['signal_physical_maxs'] = signal_physical_maxs
    # record['signal_digital_mins'] = signal_digital_mins

