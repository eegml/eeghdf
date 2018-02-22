# eeg-hdfstorage
Project to develop a easily accessible forbfor storing EEG in a way that is easy to access for machine learning.
- hdf5 based format
- looked at edf and neo formats, see NWB
- simplier than neo, but may need more of neo's strucures
- look to add fields for clinical report text
- look to add field for montages and electrode geometry

### To Do
- [x] code to write file , initial versionn is 1000
- [X] initial scripts to convert edf to eeghdf and floating point hdf5
- [ ] code to write back to edf 
- [ ] more visualization code
- [ ] add study admin code to record info (do not seem to include this now, e.g. EEG No like V17-105)
