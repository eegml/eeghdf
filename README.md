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



## Simple install for developers
- change to the desired python environment
- make sure you have git and git-lfs installed
```
git clone https://github.com/eegml/eeghdf.git 
pip install -e eeghdf
```
- or if you just want to install as a requirement into a virtual env. Put this into your requirements.txt. The repo will be cloned into ./src/eeghdf and installed
```
-e git+https://github.com/eegml/eeghdf#egg=eeghdf

```

For oldstyle setuptools install
```
python setup-dev.py develop
```

### Re-sampling 
There are many ways to resample signals. In my examples I used an approach based upon libsamplerate because it seemed to give accurate results. Depending on your
platform there are many options. Recently I have been suing pytorch based tools a lot, torchaudio has resamplinge tools and librosa is looks very impressive.

Installation will vary but on ubuntu 18.04 I did:
```
sudo apt install libsamplerate-dev
pip install git+https://github.com/cournape/samplerate/#egg=samplerate
```

Ultimately I will move the resampling code out of this repo. Maybe put it in eegml-signal

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
