import os.path as path
import eeghdf
import eeghdf.convert

## module level globals
try:
    ROOT = path.dirname(__file__)
except NameError:
    ROOT = path.curdir

ARFILE1 = path.join(ROOT,r"../data/absence_epilepsy.eeg.h5")
ARFILE2 = path.join(ROOT,r"../data/spasms.eeg.h5")
EEGFILE1 = path.normpath(ARFILE1)
EEGFILE2 = path.normpath(ARFILE2)

#print(ARFILE1)
eeg = eeghdf.Eeghdf(EEGFILE1)



def test_basic_sechdf1020_to_mne():
    raw, info, useful_channels = eeghdf.convert.sechdf1020_to_mne(eeg)

def test_basic_hdf2mne():
    raw, info, useful_channels = eeghdf.convert.hdf2mne(eeg)
