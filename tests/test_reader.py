import os.path as path

import eeghdf.reader as reader

ROOT = path.dirname(__file__)
ARFILE1 = path.join(ROOT,r"../notebooks/archive/YA2741BS_1-1+.eeghdf")
EEGFILE = path.normpath(ARFILE1)
#print(ARFILE1)

def test_reader_open():
    eeg = reader.Eeghdf(EEGFILE)
    assert eeg != None
    




eeg = reader.Eeghdf(EEGFILE)    
