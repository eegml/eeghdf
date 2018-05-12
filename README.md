# eeg-hdfstorage

Project to develop a easily accessible format for storing EEG in a way that is easy to access for machine learning.

- hdf5 based format
- looked at edf and neo formats, see NWB
- simplier than neo, but may need more of neo's strucures
- compare with fif format of mne project to evolve
- look to add fields for clinical report text
- look to add field for montages and electrode geometry

## To Do

- [x] code to write file, target initial release version is 1000
- [X] initial scripts to convert edf to eeghdf and floating point hdf5
- [ ] code to subsample and convert edf -> eeghdf
- [ ] code to write back to edf
- [ ] more visualization code -> push to eegvis
- [ ] add study admin code to record info (do not seem to include this now, e.g. EEG No like V17-105)
- [ ] code to clip and create subfiles
  - [ ] allow patient info to propate
  - [ ] hash list/tree of history of file so that can track provenance of waveforms if desired
  - [ ] clip and maintain correct (relative) times
