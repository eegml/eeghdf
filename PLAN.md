## To Do

- [x] code to write file, target initial release version is 1000
- [X] initial scripts to convert edf to eeghdf and floating point hdf5
- [x] code to subsample and convert edf -> eeghdf
- [ ] code to write back to edf
- [x] more visualization code -> push to eegvis
- [x] add convenience interface to phys_signal with automagic conversion from digital->phys units
      - should this use a subclass of numpy?
- [ ] add study admin code to record info (do not seem to include this now, e.g. EEG No like V17-105)
- [ ] code to clip and create subfiles
  - [ ] allow patient info to propagate
  - [ ] hash list/tree of history of file so that can track provenance of waveforms if desired
  - [ ] clip and maintain correct (relative) times
- [ ] consider how to handle derived records: for example the downsampled float32 records "frecord200Hz" 

- [x] added prototype support for S3 read-only support (v 0.2.4)
- [x] 2022-jul-19: removed reference to resampling code in README
