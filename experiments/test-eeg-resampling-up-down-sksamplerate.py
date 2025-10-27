# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python [conda env:anaconda3]
#     language: python
#     name: conda-env-anaconda3-py
# ---

# %%
# # %load eeg_resampling.py
# %%
# %matplotlib inline
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

import scikits.samplerate as sk_samplerate

import eeghdf
import eegvis.stacklineplot as stackplot
# %%

# %%
# check versions
print("scikits.samplerate (Secret Rabbit code) version:", sk_samplerate.__version__)
print("scipy:", scipy.__version__)
print("matplotlib.__version__")
# %%
plt.rcParams["figure.figsize"] = (24, 9)

# %%

# %%
hf = eeghdf.Eeghdf("../../eeghdf/data/absence_epilepsy.eeghdf")
print("original shape:", hf.phys_signals.shape)

eegsig = hf.phys_signals[0:30, 0:100000]
eegsigt = eegsig.transpose()
# %%

fs0 = hf.sample_frequency  # usually 200
fs1 = 156
fs2 = 100
fs3 = 50

A = 0
B = 3

# %%
# %%
stackplot.stackplot_t(eegsigt[0 : int(20000), A:B])

# %%
# %% Cell[]
eegdownt1 = sk_samplerate.resample(eegsigt, fs1 / fs0, "sinc_best", verbose=True)
eegdownt2 = sk_samplerate.resample(eegsigt, fs2 / fs0, "sinc_best", verbose=True)

print("eegdownt1.shape:", eegdownt1.shape)

# %%
# %%
stackplot.stackplot_t(eegdownt1[0 : int(20000 * (fs1 / fs0)), A:B])
# %%

# %%
stackplot.stackplot_t(eegdownt2[0 : int(20000 * (fs2 / fs0)), A:B])

# %%
# %%
eegdownt3 = sk_samplerate.resample(eegsigt, fs3 / fs0, "sinc_best")
stackplot.stackplot_t(eegdownt3[0 : int(20000 * (fs3 / fs0)), A:B])
print("ratio:", fs3 / fs0)

# %%
# %%
fs4 = 15
eegdownt4 = sk_samplerate.resample(eegsigt, fs4 / fs0, "sinc_best")
stackplot.stackplot_t(eegdownt4[0 : int(20000 * (fs4 / fs0)), A:B])
# %%

# %%
fs5 = 10
eegdownt5 = sk_samplerate.resample(eegsigt, fs5 / fs0, "sinc_best")
stackplot.stackplot_t(eegdownt5[0 : int(20000 * (fs5 / fs0)), A:B])
# %%

# %%
eegupt3 = sk_samplerate.resample(eegdownt3, fs0 / fs3, "sinc_best")

stackplot.stackplot_t(eegupt3[0:20000, A:B])

# %%
print(
    "ratio:",
    fs0 / fs3,
    "down/upsample shape:",
    eegupt3.shape,
    "original shape:",
    eegsigt.shape,
)

# %%
stackplot.stackplot_t(eegsigt[0:20000, A:B])

# %%
sum(eegsigt[0:20000, 0])

# %%
sum(eegupt3[0:20000, 0])

# %%
sum(np.abs(eegupt3[0:20000, 0])) - sum(np.abs(eegsigt[0:20000, 0]))

# %%
dd = eegsigt[0:20000, 0] - eegupt3[0:20000, 0]

# %%

# %%
V = np.var(eegsigt[0:20000, 0])

# %%
print("Variance:", V, "Stderr:", math.sqrt(V))

# %%
plt.plot(dd / math.sqrt(V))
plt.title("error relative to stderr of original signal")
plt.show()

# %%
import seaborn

seaborn.set_style("whitegrid")
ch = 1
t1, t2 = 0, 2000
plt.plot(eegsigt[t1:t2, ch], color="black", alpha=0.5, label="original %s" % fs0)
plt.plot(eegupt3[t1:t2, ch], color="red", alpha=0.5, label="downsampled %s" % fs3)
plt.legend()
plt.title("compare original and subsampled/upsampled signal")
plt.show()

# %%
t1, t2 = 2000, 4000
plt.plot(eegsigt[t1:t2, ch], color="red", alpha=0.5, label="original %s" % fs0)
plt.plot(eegupt3[t1:t2, ch], color="black", alpha=0.5, label="downsampled %s" % fs3)
plt.legend()
plt.title("compare original and subsampled/upsampled signal")
plt.show()

# %%
# now let's see how it looks with edf files
import edflib
import os.path

# %%
TUH_SZ_ROOT = "/mnt/data1/eegdbs/TUH/temple/tuh-sz-v1.2.0/v1.2.0"
tuhedf_fn = os.path.join(
    TUH_SZ_ROOT, "eval/01_tcp_ar/00006059/s003_2012_05_25/00006059_s003_t000.edf"
)
tuhedf_fn = "../../eeghdf/data/00000115_s07_a01.edf"

# %%
print(tuhedf_fn)
ef = edflib.EdfReader(tuhedf_fn)

# %%
N = 27
[ef.samplefrequency(ch) for ch in range(N)]

# %%
ef.get_signal_text_labels()

# %%
ef.get_samples_per_signal()

# %%
num_samples = ef.get_samples_per_signal()[0]  # same for 0:27
num_samples

# %%
edf_orig = np.zeros((27, num_samples), dtype="float32")

# %%
edf_orig[0, :] = ef.get_signal(0)
# for ch in range(N):
#   edf_orig[ch,:] = ef.get_signal(ch)
effs0 = ef.samplefrequency(0)

# %%

# %%
for ch in range(1, 26):
    edf_orig[ch, :] = ef.get_signal(ch)

# %%
stackplot.stackplot(edf_orig[:, 0:2500])

# %%
edf_origt = edf_orig.transpose()
print("edf_origt.shape", edf_origt.shape)

# %%
# %time edf_down200t = sk_samplerate.resample(edf_origt, 200/effs0, 'sinc_best')

# %%
edf_down200t.shape

# %%
stackplot.stackplot_t(edf_down200t[0:2000, :])

# %%
edf_downupt = sk_samplerate.resample(edf_down200t, effs0 / 200, "sinc_best")

# %%
edf_downupt.shape

# %%
ch = 18
s1, s2 = 0, 2500  # ten seconds
plt.plot(edf_orig[ch, 0:2500], color="black", alpha=0.5)
plt.plot(edf_downupt[0:2500, ch], color="red", alpha=0.5)
plt.show()

# %%
ch = 5
t1, t2 = 40, 42
s1, s2 = t1 * 250, t2 * 250
plt.plot(edf_orig[ch, s1:s2], color="black", alpha=0.5, label="orignal")
plt.plot(edf_downupt[s1:s2, ch], color="red", alpha=0.5, label="resampled")
plt.title("%s seconds (%.0f-%.0f) of channel %s" % (t2 - t1, t1, t2, ch))
plt.legend()
plt.show()

# %%
# now try big all at once resample
big = hf.phys_signals[:27, :]
big.shape

# %%
# %time bigdown = sk_samplerate.resample(big.transpose(), 100/200, 'sinc_best')

# %%
bigdown.shape

# %%

# %%
import resampy

# %%
res0 = effs0
res1 = 150
# %time resbigdown = resampy.resample(big, res0,res1, axis=1, filter='kaiser_best')

# %%
resbigdownup = resampy.resample(resbigdown, res1, res0, axis=1, filter="kaiser_best")

# %%
print("big.shape:", big.shape)
print(resbigdown.shape)
print(resbigdownup.shape)

# %%
ch = 5
t1, t2 = 40, 42
s1, s2 = t1 * 250, t2 * 250
plt.plot(big[ch, s1:s2], color="black", alpha=0.5, label="orignal")
plt.plot(resbigdownup[ch, s1:s2], color="red", alpha=0.5, label="resampy resampled")
plt.title("%s seconds (%.0f-%.0f) of channel %s" % (t2 - t1, t1, t2, ch))
plt.legend()
plt.show()  # add show() and it will display in gitlab

# %%
# note with resampy I am using floating point arrays as input and I am setting the axis=1 argument with this shape of input arrays

# %%
# haven't tried using integer samples yet
# was seeing artifacts with resampy

# test scipy.resample and compare.

# leave hdf5 files 00000_s0000_.edf  -> 00000_s0000.f200.eeg.h5
# Stanford data set convert to physical signals
