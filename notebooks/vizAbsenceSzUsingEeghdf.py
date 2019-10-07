# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os

try:
    os.chdir(os.path.join(os.getcwd(), "notebooks"))
    print(os.getcwd())
except:
    pass
#%%
from IPython import get_ipython

#%% [markdown]
# ## Introduction to visualizing data in the eeghdf files
#%% [markdown]
# ### Getting started
# The EEG is stored in hierachical data format (HDF5). This format is widely used, open, and supported in many languages, e.g., matlab, R, python, C, etc.
#
# Here, I will use the eeghdf library in python for more convenient access than using raw h5py

#%%
# import libraries
from __future__ import print_function, division, unicode_literals

get_ipython().run_line_magic("matplotlib", "inline")
# %matplotlib notebook # allows interactions

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import eeghdf
from pprint import pprint

import stacklineplot  # local copy of eegvis.stacklineplot


#%%
# Make all the figures bigger and easier to see in this notebook
# matplotlib.rcParams['figure.figsize'] = (18.0, 12.0)
FIGSIZE = (12.0, 8.0)  # use with %matplotlib inline
matplotlib.rcParams["figure.figsize"] = FIGSIZE

#%% [markdown]
# ### Direct access via h5py library
# We have written a helper library eeghdf to conveniently access these hdf5 files.
# Note but you are not required to use this as you can access all the data via hdf5 libraries.

#%%
# first open the hdf5 file
eegf = eeghdf.Eeghdf("../data/absence_epilepsy.eeghdf")

# show the groups at the root of the tree as a list

#%% [markdown]
# We can focus on the patient group and access it via hdf['patient'] as if it was a python dictionary. Here are the key,value pairs in that group. Note that the patient information has been anonymized. Everyone is given the same set of birthdays. This shows that this file is for Subject 2619, who is male.

#%%
# here is some basic info
print(f"eegf.file_name: {eegf.file_name}")
print(f"eegf.age_years: {eegf.age_years}")

print(f"eegf.number_channels: {eegf.number_channels}")
print(f"sample_frquency: {eegf.sample_frequency}")
print(f"eegf.patient: {eegf.patient}")
print(f"eegf.start_isodatetime: {eegf.start_isodatetime}")
print(f"eegf.end_isodatetime: {eegf.end_isodatetime}")
#%%
# can get this list of electrode labels
print(f"eegf.electrode_labels: {eegf.electrode_labels}")
#%%
# the underlying hdf5 file handle (from h5py) is available at
eegf.hdf

#%% [markdown]
# Now we look at how the waveform data is stored. By convention, the first record is called "record-0" and it contains the waveform data as well as the approximate time (relative to the birthdate)at which the study was done, as well as technical information like the number of channels, electrode names and sample rate.
# It can be accessed as the sampled numbers by using eegf.rawsignals[ch,sample]
# (usually int16 or int32)
#%%
eegf.rawsignals[
    4, 2000:2020
]  # get channel 4, samples 2000 to 2020 (remember start counting at zero)
plt.plot(eegf.rawsignals[4, 2000:2020])

#%% [markdown]
# often you will instead want the signals converted/scaled into physically meaniful units. For EEG this will usually
# be micro volts (uV) or possibly mV
eegf.phys_signals[4, 2000:2020]

#%%
t = np.arange(2000 / 200, 2020 / 200, step=1 / 200)
plt.plot(t, eegf.phys_signals[4, 2000:2020])
plt.ylabel("uV")
plt.xlabel("sec")
#%%

#%%

#%% [markdown]
# We can also grab the actual waveform data and visualize it. Using the helper library for matplotlib stackplot.py.
#
#
# [More work is being done in the eegml-eegvis package for more sophisticated visualization.]

#%%
electrode_labels = eegf.shortcut_elabels
# numbered_electrode_labels = ["%d:%s" % (ii, str(labels[ii], 'ascii')) for ii in range(len(labels))]

#%% [markdown]
# #### Simple visualization of EEG (brief absence seizure)

#%%
# choose a point in the waveform to show a seizure
stacklineplot.show_epoch_centered(
    eegf.phys_signals,
    1476,
    epoch_width_sec=15,
    chstart=0,
    chstop=19,
    fs=eegf.sample_frequency,
    ylabels=eegf.shortcut_elabels,
    yscale=3.0,
)
plt.title("Absence Seizure")

#%% [markdown]
# ### Annotations
# It was not a coincidence that I chose this time in the record. I used the annotations to focus on portion of the record which was marked as having a seizure.
#
# You can access the clinical annotations.
# If you need them, you can get the raw list using eegf._annotation_text and eegf._annotation_start100ns
# They are originally stored with integer start times counting in unitsl of 100ns from the beginning of the file
#%%
# here is a sample
list(zip(eegf._annotation_text, eegf._annotation_start100ns))[:10]


#%% [markdown]
# But usually, I have accessed them by requesting them as a dataframe
eegf.edf_annotations_df

#%% [markdown]
# It is easy then to find the annotations related to seizures

#%%
eegf.annotations_contain("sz", case=False)


#%%
print("matplotlib.__version__:", matplotlib.__version__)
print("eeghdf.__version__", eeghdf.__version__)
print("pandas.__version__:", pd.__version__)


#%%

