import os.path as path
import numpy as np

import eeghdf.reader as reader

ROOT = path.dirname(__file__)
ARFILE1 = path.join(ROOT,r"../data/absence_epilepsy.eeghdf")
ARFILE2 = path.join(ROOT,r"../data/spasms.eeghdf")
EEGFILE1 = path.normpath(ARFILE1)
EEGFILE2 = path.normpath(ARFILE2)

#print(ARFILE1)
eeg = reader.Eeghdf(EEGFILE1)    
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

def test_phys_signals1():
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

    selected_channels = [4,1,7] # out of order indexing
    res = eeg.phys_signals[selected_channels,x*200:x*200+200]
    assert res.shape == (3, 200)


    


def test_phys_signals_test_multi_dimslice():
    '''test PhysicalSignalZeroOffset (slice, slice)
    indexing for zero-offset physSignal object'''
    x = 1420
    print(eeg.phys_signals.shape)
    res = eeg.phys_signals[1:3, x*200:x*200+5] #     eeg.phys_signals[1:3, x*200:x*200+5]

    target = np.array([[ -69.53124627, -158.49608524, -152.05077308, -168.16405347,
                         -210.74217618],
                       [ -99.60936965,  -78.02733956,   -7.81249958,  -10.83984317,
                         -35.83984182]])
    assert np.all(res - target < 0.1)


def test_phys_signals_test_slice_int():
    '''test PhysicalSignalZeroOffset (slice, int)
    indexing for zero-offset physSignal object'''
    x = 1420

    res = eeg.phys_signals[1:3, 5] 
    print('test_phys_signals_test_slice_int: res.shape:', res.shape)
    target = np.array([[ 49.99999731],
                       [ 49.99999731] ])
                      
    assert np.all(res - target < 0.1)

def test_phys_signals_test_int_slice():
    """
    test PhysicalSignalZeroOffset (int, slice)
    indexing for zero-offset physSignal object'''
    returns a 1D array 
    """

    res = eeg.phys_signals[5,2000:2005]
    assert res.shape == (5,)
    target = np.array([-19.92187393, -15.52734292, -10.54687443,  -6.64062464,  -8.78906203])
    assert np.all(res - target < 0.1 )


def test_phys_signals_test_fancy_channel_index():

    chsel_3 = [1,5,7]
    chsel_4 = [1,3,5,7]
    target = np.array([[-139.35546126, -140.23436747, -136.32811768, -131.15233671, -125.29296202],
                       [   2.34374987,    4.00390603,    7.91015583,   12.89062431, 14.06249924],
                       [ -19.92187393,  -15.52734292,  -10.54687443,   -6.64062464, -8.78906203],
                       [  64.64843403,   67.18749639,   67.18749639,   68.06640259, 70.11718373]])
    res = eeg.phys_signals[chsel_4,2000:2005]
    assert np.all(target - res < 0.1)

    # note that using list will always get a matrix back, not a row vector like with an int 
    chsel_1 = [5]
    res = eeg.phys_signals[chsel_1, 2000:2005]
    tar = np.array([[-19.92187393, -15.52734292, -10.54687443,  -6.64062464, -8.78906203]])
    assert np.all(res-tar < 0.1)
    assert res.shape == (1,5)

# assume original shape at least (4,10)
indexing2D_test_fancy_list1 = [1,3,4]
indexing2D_test_fancy_list2 = [2]
indexing2D_tests_constrained = [
    (slice(0,2), slice(0,10)), # arr[0:2,0:10]
    (1, slice(2,4)), # arr[1,2:4]
    (slice(2,4), 1), # arr[2:4,1]
    (3,2), # arr[3,2]
    (indexing2D_test_fancy_list1, slice(0,9)),
    (indexing2D_test_fancy_list2, slice(0,9)),
    (slice(0,3), indexing2D_test_fancy_list1),
    (slice(0,3), indexing2D_test_fancy_list2),
    ]
# slices with full channels 
indexing2D_tests_unconstrained = [(slice(None,None,None), slice(2,3)), # [:, 2:3]
                                  (slice(None,None,None), slice(2,4))] # [:, 2:4]


    
def test_phys_signals_indexing():
    x = 1420
    eeg.phys_signals[0:2, 0:200]
    print( 'eeg.phys_signals[3,4]:',  eeg.phys_signals[3,4])
    assert abs(eeg.phys_signals[3,4] - 49.9999973144) < 0.01 
    print('eeg.phys_signals[3,400]:', eeg.phys_signals[3,400])
    assert abs(eeg.phys_signals[3,400] ) < 0.1 # test calibration 50uV

    eeg.phys_signals[3:5]

    selected_channels = [1,4,7] # test fancy indexing of channels
    res = eeg.phys_signals[selected_channels,x*200:x*200+200]
    print(res.shape)
    assert res.shape == (3, 200)

    selected_channels = [4,1,7] # out of order indexing
    res = eeg.phys_signals[selected_channels,x*200:x*200+200]
    print('test phys_signals res.shape:', res.shape)
    assert res.shape == (3, 200)

def test_eeghdf_ver2():
    # open the old file version
    eeg = reader.Eeghdf_ver2(EEGFILE1)
    assert eeg != None
    print(eeg.hdf.attrs['EEGHDFversion'])


# def test_s2u_diag_broadcasting():

num_chan = 5
S = np.arange(num_chan*10, dtype='float64')
    # array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
    #     11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
    #     22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.])
S.shape = (num_chan, 10) # reshape to emulate an EEG signal 

sca = np.arange(float(num_chan))
transform = S * sca[:,np.newaxis]
# assert np.all(target == transform)

def test_all_indexing():
    for ss in indexing2D_tests_constrained:
        res = eeg.phys_signals[ss]
        tar = S[ss]
        print(res.shape, tar.shape)
        assert np.all(res.shape == tar.shape)
    
def test_indexing_unconstrained():
    for ss in indexing2D_tests_unconstrained:
        compare = S[ss] # has limited numbers
        res = eeg.phys_signals[ss]
        assert res.shape[0] == eeg.number_channels
