
### mne fif format "Functional Image file format" (FIFF)
- why we use FIF question from a Chris Holdgraf in 2018
https://github.com/mne-tools/mne-python/issues/5302

excerpt from that thread
>>>
FIF is a format that was borne out of the Neuromag system and it is well-structured to be maximally compatible with all the different physiological signals. The format architecture has been primarily driven by Matti H. and Elekta IIRC. It serves as a great format but it has primarily targeted to MEG data so it doesn't have as broad of a user base as the EDF format. It also has a historic max size of 2GB but that can be compensated for in linking files.
