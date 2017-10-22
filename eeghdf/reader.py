# -*- coding: utf-8 -*-
"""
functions to help with reading eeghdf files
versions 1...
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

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
    def __init__(self, fn):
        self.file_name = fn
        self.hdf = h5py.File(fn)
        self.rec0 = self.hdf['record-0']
        rec0 = self.rec0
        self.age_years = rec0.attrs['patient_age_days']/365
        
        self.rawsignals = rec0['signals']
        labels_bytes = rec0['signal_labels']
        self.electrode_labels = [str(s,'ascii') for s in labels_bytes]

        # read annotations and format them for easy use
        annot = rec['edf_annotations'] # but these are in a funny 3 array format
        antext = [s.decode('utf-8') for s in annot['texts'][:]]
        self._annotation_text = antext
        starts100ns = [xx for xx in annot['starts_100ns'][:]]
        self._annotation_start100s = starts100ns
        start_time_sec = [xx/10**7 for xx in starts 100s] # 10**7 * 100ns = 1sec
        df = pd.DataFrame(data=antext,columns=['text'])
        df['starts_sec'] = start_time_sec
        df['starts_100ns'] = starts100ns
        self.edf_annotations_df = df

        # what about units and conversion factors
        # record.attrs['start_isodatetime'] = start_isodatetime
        # record.attrs['end_isodatetime'] = end_isodatetime

        # record.attrs['number_channels'] = number_channels
        # record.attrs['number_samples_per_channel'] = num_samples_per_channel
        # record.attrs['sample_frequency'] = sample_frequency

        # record['signal_physical_mins'] = signal_physical_mins
        # record['signal_physical_maxs'] = signal_physical_maxs
        # record['signal_digital_mins'] = signal_digital_mins
        # record['signal_digital_maxs'] = signal_digital_maxs
        # record.attrs['patient_age_days'] = patient_age_days
        # record.attrs['technician'] = technician
        # patient
        self.patient = dict(self.hdf['patient'].attrs)

    def annotations_contain(self, pat, case=False):
        df = self.edf_annotations_df
        return df[df.text.str.contains(pat,case=case)]
