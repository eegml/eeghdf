# eeghdf

Project to develop a easily accessible format for storing EEG in a way that is easy to access for machine learning.

### Features
Features derived from hdf5 format:
- hdf5 offers reliable, checksummed and compressed storage of digital EEG which was designed for long-term storage of data
- hdf5 is supported widely C, C++, javascript, python, julia, matlab, 
- eeghdf offers a numpy-like interface to data without requiring the whole file to be loaded in memory
- efficient reading (the whole file is not read into memory to access data)
- cloud enabled direct streaming from S3 buckets via the rcos3 driver
- "self documenting" and extensible
- advanced features: parallel readers/single writer, MPI, streaming supported

Additional goals/features:
- build set of tools to visualize and analyze EEG based upon this format, visualization
- easy convertion to other formats: first target is mne-python "raw" format, BIDS-EEG next?

### Alternatives, background research and future goals
  
- looked at edf and neo formats, see [Neurodata Without Borders](https://github.com/NeurodataWithoutBorders). Compare with [XDF](https://github.com/sccn/xdf/).
  - simplier than neo, but may need more of neo's structures as use grows
- [ONE format](https://int-brain-lab.github.io/ONE/one_reference.html)
- compare with [MNE](http://martinos.org/mne/stable/index.html) fif format of mne project to evolve


##### future goals
- look to support multiple records and different sampling rates
- look to add fields for clinical report text
- look to add field for montages and electrode geometry
- "extension" group

## installation
```
pip install eeghdf
```

#### Simple install for developers
This assumes you want to make changes to the eeghdf code.
- change to the desired python virtual environment
- make sure you have git and git-lfs installed
```
git clone https://github.com/eegml/eeghdf.git 
cd eeghdf

python setup-dev.py develop
```

