# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:56:31 2018

@author: skjerns

Gist to save a mne.io.Raw object to an EDF file using pyEDFlib
(https://github.com/holgern/pyedflib)

Disclaimer:
    - Saving your data this way will result in slight 
      loss of precision (magnitude +-1e-09).
    - It is assumed that the data is presented in Volt (V), it will be internally converted to microvolt
    - Saving to BDF can be done by changing the file_type variable.
      Be aware that you also need to change the dmin and dmax to
      the corresponding minimum and maximum integer values of the
      file_type: e.g. BDF+ dmin, dmax =- [-8388608, 8388607]
"""

import pyedflib # pip install pyedflib
from datetime import datetime
import mne
import os


def write_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+ filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk

    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
    # static settings
    file_type = pyedflib.FILETYPE_EDFPLUS 
    sfreq = mne_raw.info['sfreq']
    date = datetime.now().strftime( '%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None

    
    # convert data
    channels = mne_raw.get_data(picks, 
                                start = first_sample,
                                stop  = last_sample)
    
    # convert to microvolts to scale up precision
    channels *= 1e6
    
    # set conversion parameters
    dmin, dmax = [-32768,  32767]
    pmin, pmax = [channels.min(), channels.max()]
    n_channels = len(channels)
    
    # create channel from this   
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels, 
                               file_type=file_type)
        
        channel_info = []
        data_list = []
        
        for i in range(n_channels):
            ch_dict = {'label': mne_raw.ch_names[i], 
                       'dimension': 'uV', 
                       'sample_rate': sfreq, 
                       'physical_min': pmin, 
                       'physical_max': pmax, 
                       'digital_min':  dmin, 
                       'digital_max':  dmax, 
                       'transducer': '', 
                       'prefilter': ''}
        
            channel_info.append(ch_dict)
            data_list.append(channels[i])
        
        f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(data_list)
    except Exception as e:
        print(e)
        return False
    finally:
        f.close()    
    return True

if __name__ == '__main__':
    import sys
    import os.path

    if len(sys.argv) == 2:
        fn = sys.argv[1]
        base, ext = os.path.splitext(fn)
        raw = mne.io.read_raw_fif(fn)
        write_edf(raw, base+'.edf')
        
