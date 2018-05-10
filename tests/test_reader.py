import os.path as path
import numpy as np

import eeghdf.reader as reader

ROOT = path.dirname(__file__)
ARFILE1 = path.join(ROOT,r"../data/absence_epilepsy.eeghdf")
ARFILE2 = path.join(ROOT,r"../data/spasms.eeghdf")
EEGFILE1 = path.normpath(ARFILE1)
EEGFILE2 = path.normpath(ARFILE2)

#print(ARFILE1)

def test_reader_open():
    eeg = reader.Eeghdf(EEGFILE1)
    assert eeg != None


def test_reader_duration():
    eeg = reader.Eeghdf(EEGFILE2)
    dur = eeg.duration_seconds
    calc_dur = eeg.number_samples_per_channel / eeg.sample_frequency
    check_val = dur - calc_dur # example: 446000/200 = 2230

    print('dur:', dur, 'calc_dur:', calc_dur)
    assert check_val*check_val < 1.0
    
    

def test_calc_sample_units():
    eeg = reader.Eeghdf(EEGFILE1)
    eeg._calc_sample2units()
    assert np.all(eeg._s2u)

def test_min_maxes():
    eeg = reader.Eeghdf(EEGFILE1)
    assert np.all(eeg.signal_physical_mins)
    assert np.all(eeg.signal_physical_maxs)

    assert np.all(eeg.signal_digital_maxs == np.array([32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
       32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
       32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
       32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767]))

    assert np.all(eeg.signal_digital_mins == np.array([-32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
       -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
       -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
       -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
       -32768, -32768, -32768, -32768]))

def test_phys_signals():
    '''mostly just a smoke test of zero-offset physSignal object'''
    x = 1420
    eeg.phys_signals[:, x * 200:x * 200 + 10 * 200]
    print(eeg.phys_signals.shape)
    eeg.phys_signals[4, 0:200]
    print( 'eeg.phys_signals[3,4]:',  eeg.phys_signals[3,4])
    assert abs(eeg.phys_signals[3,4] - 49.9999973144) < 0.01 
    print('eeg.phys_signals[3,400]:', eeg.phys_signals[3,400])
    assert abs(eeg.phys_signals[3,400] ) < 0.1 # test calibration 50uV

    eeg.phys_signals[3:5]

    selected_channels = [1,4,7] # test fancy indexing of channels
    res = eeg.phys_signals[selected_channels,x*200:x*200+200]
    print(res.shape)
    assert res.shape == (3, 200)

eeg = reader.Eeghdf(EEGFILE1)    

def test_eeghdf_ver2():
    # open the old file version
    eeg = reader.Eeghdf_ver2(EEGFILE1)
    assert eeg != None
    print(eeg.hdf.attrs['EEGHDFversion'])
