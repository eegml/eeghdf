# eeghdf

Project to develop a easily accessible format for storing EEG in a way that is easy to access for machine learning.

- hdf5 based format
- looked at edf and neo formats, see [Neurodata Without Borders](https://github.com/NeurodataWithoutBorders)
- simplier than neo, but may need more of neo's structures
- compare with [MNE](http://martinos.org/mne/stable/index.html) fif format of mne project to evolve
- look to add fields for clinical report text
- look to add field for montages and electrode geometry


## Simple install for developers
- change to the desired python environment
```
git clone https://github.com/eegml/eeghdf.git 
pip install -e eeghdf
```
- or if you just want to install as a requirement into a virtual env. Put this into your requirements.txt. The repo will be cloned into ./src/eeghdf and installed
```
-e git+https://github.com/eegml/eeghdf#egg=eeghdf
```
## To Do

- [x] code to write file, target initial release version is 1000
- [X] initial scripts to convert edf to eeghdf and floating point hdf5
- [x] code to subsample and convert edf -> eeghdf
- [ ] code to write back to edf
- [ ] more visualization code -> push to eegvis
- [x] add convenience interface to phys_signal with automagic conversion from digital->phys units
- [ ] add study admin code to record info (do not seem to include this now, e.g. EEG No like V17-105)
- [ ] code to clip and create subfiles
  - [ ] allow patient info to propagate
  - [ ] hash list/tree of history of file so that can track provenance of waveforms if desired
  - [ ] clip and maintain correct (relative) times
- [ ] consider how to handle derived records: for example the downsampled float32 records "frecord200Hz" 
