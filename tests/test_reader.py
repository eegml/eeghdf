import os.path as path
import numpy as np

import eeghdf.reader as reader

ROOT = path.dirname(__file__)
ARFILE1 = path.join(ROOT,r"../notebooks/archive/YA2741BS_1-1+.eeghdf")
EEGFILE = path.normpath(ARFILE1)
#print(ARFILE1)

def test_reader_open():
    eeg = reader.Eeghdf(EEGFILE)
    assert eeg != None
    

def test_calc_sample_units():
    eeg = reader.Eeghdf(EEGFILE)
    eeg._calc_sample2units()
    assert np.all(eeg._s2u)

def test_min_maxes():
    eeg = reader.Eeghdf(EEGFILE)
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
    assert eeg.phys_signals[3,4] == 0.0
    assert abs(eeg.phys_signals[3,400] - 50.0 ) < 0.1 # test calibration 50uV

    eeg.phys_signals[3:5]

eeg = reader.Eeghdf(EEGFILE)    
