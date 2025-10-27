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

# %% [markdown]
# I am using this to test out the downsampling code
# and document how to use the montage viewing.
# -Chris
# 2018-05-07

# %% nbpresent={"id": "2420c58b-7e66-4a7f-9cc3-a21f59b2ee8b"}
# %matplotlib inline

# %% nbpresent={"id": "b26880c1-2e47-4ac3-aeec-cdbdc43df65b"}
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16, 9)

# %% nbpresent={"id": "48f32c3a-8e63-4a0a-b6e1-85414a0c95f2"}
import eegvis
import eegvis.stackplot_bokeh as sbplot
import eegvis.stacklineplot as splot
import eegvis.montageview

# %% nbpresent={"id": "071e407e-131a-46d4-b6de-ec6a0f18f055"}
import eeghdf


# %% nbpresent={"id": "39867397-86a7-4644-80a8-aade4a60a1c2"}
# ls ../data

# %% nbpresent={"id": "897e42b6-14f8-4467-8ea5-ee20920f3c5b"}
testfile = "../data/tuh_00000115_s07_a00_f200.eeg.h5"
testfile2 = "../data/absence_epilepsy.eeghdf"
testfile3 = "../data/spasms.eeghdf"
# hf = h5py.File(testfile, 'r')
eegf = eeghdf.Eeghdf_ver2(testfile)
hf = eegf.hdf
eegf2 = eeghdf.Eeghdf(testfile2)
eegf3 = eeghdf.Eeghdf(testfile3)

# %% nbpresent={"id": "e4b4328c-6d44-4191-9cf1-986eb8445529"}
phys_signals = eegf2.phys_signals

# %% nbpresent={"id": "8faa8806-7590-4990-a280-689bec01da44"}
elabels = eegf.electrode_labels

# %% nbpresent={"id": "3065018b-9133-4130-8d4b-527dc43c2b62"}
phys_signals

# %% [markdown]
# ### Verifying the scaling of the signals
# On our Nihon-Koden amplifiers, it is standard to begin the recording with a 50 uV square wave signal. From wht I can tell, it looks like a 50uV positive square wave.

# %%

plt.plot(phys_signals[5, 0 : int(8 * eegf2.sample_frequency)])
plt.title("channel 5, first 8s, sample frequency %s" % eegf2.sample_frequency)
plt.xlabel("sample number")
plt.ylabel("uV")
plt.show()

# %%
ch = 10
width = 10
start = 120
fs = eegf.sample_frequency
plt.plot(eegf.phys_signals[ch, int(start * fs) : int(fs * (start + width))])
plt.title("channel 5, first %ss, sample frequency %s" % (width, eegf2.sample_frequency))
plt.xlabel("sample number")
plt.ylabel("uV")
plt.show()

# %%
import scipy.stats

# %%
print("%s channel" % eegf.electrode_labels[ch])
print(scipy.stats.describe(eegf.phys_signals[ch, :]))

# %%
print("%s channel" % eegf2.electrode_labels[ch])
print(scipy.stats.describe(eegf2.phys_signals[ch, :]))

# %%
# np.histogram(eegf2.phys_signals[ch,:],bins='auto')
plt.hist(eegf2.phys_signals[ch, :], bins="auto", label=eegf2.file_name)
plt.hist(eegf.phys_signals[ch, :], bins="auto", label=eegf.file_name)
plt.hist(
    eegf3.phys_signals[ch, :] - 1000,
    bins="auto",
    label=eegf3.file_name + " shifted -1000",
)
plt.legend()
plt.title("distribution of values for SEC and TUH")
plt.show()

# %%
print("ages:", eegf.age_years, eegf2.age_years, eegf3.age_years)

# %%
eegf2._s2u

# %% nbpresent={"id": "5bf69369-2bf2-4706-9e04-a0a0da3c3a5c"}
splot.stackplot(phys_signals[0:19:, 0:2000], ylabels=elabels, yscale=3.0)

# %% nbpresent={"id": "0f90188f-afcb-4fad-8b99-afaac7cbfab7"}
splot.show_epoch_centered(
    phys_signals,
    goto_sec=5,
    epoch_width_sec=10,
    chstart=0,
    chstop=21,
    fs=200,
    ylabels=elabels,
)

# %% nbpresent={"id": "6c1b0f6d-6930-461e-a323-58e69d636738"}
splot.show_epoch_centered(
    phys_signals,
    goto_sec=15,
    epoch_width_sec=10,
    chstart=0,
    chstop=21,
    fs=200,
    ylabels=elabels,
)

# %% nbpresent={"id": "31d0215a-04eb-44e8-b18c-0ee185fc50f6"}
splot.show_epoch_centered(
    phys_signals,
    goto_sec=25,
    epoch_width_sec=10,
    chstart=0,
    chstop=21,
    fs=200,
    ylabels=elabels,
    yscale=1.5,
)

# %% nbpresent={"id": "6f2244c4-3da1-4385-84ca-dbf647744280"}
eegvis.montageview.DB_LABELS

# %% nbpresent={"id": "f757b87f-c6e9-4211-a4a1-03a1a8b27bc1"}
elabels

# %% nbpresent={"id": "6cb1954f-7dc2-48bd-91b0-c0d2635e27b1"}
rlabels = eegvis.montageview.standard2shortname(elabels)
rlabels

# %% nbpresent={"id": "3d536473-6389-477d-8689-2bfa47ed1abb"}
# make standard, wonder if I should make all uppercase?
# FP1 -> Fp1, FP2 -> Fp2, CZ -> Cz necessary to for double banana
replacement_dict = {"FP1": "Fp1", "FP2": "Fp2", "CZ": "Cz", "PZ": "Pz", "FZ": "Fz"}
rlabels = [eegvis.montageview.replace_all(text, replacement_dict) for text in rlabels]


# %% nbpresent={"id": "11421eca-17cf-4ef4-8405-9fe66a1b9a8d"}
monv = eegvis.montageview.MontageView(eegvis.montageview.DB_LABELS, rlabels)

# %% nbpresent={"id": "d0ee571a-9336-4d53-b644-3ecc899c134a"}
V = eegvis.montageview.double_banana_set_matrix(monv.V)

# %% nbpresent={"id": "60fd7615-006c-4cda-a978-1efdd01fe6a1"}
splot.show_montage_centered(
    phys_signals,
    monv,
    15,
    epoch_width_sec=10,
    chstart=0,
    chstop=21,
    fs=200,
    ylabels=elabels,
    yscale=2.0,
)

# %% nbpresent={"id": "73a81270-4c93-40a1-87b8-17201a2fe68f"}
# now reverse for clinical use
monv.V = -monv.V

# %% nbpresent={"id": "c3a4295a-2ead-4261-ab10-5728f5c71b54"}
splot.show_montage_centered(
    phys_signals,
    monv,
    15,
    epoch_width_sec=10,
    chstart=0,
    chstop=21,
    fs=200,
    ylabels=rlabels,
    yscale=2.0,
)

# %%
# now let's compare with the original waveform
signals = eegf.rawsignals
splot.show_montage_centered(
    signals,
    monv,
    15,
    epoch_width_sec=10,
    chstart=0,
    chstop=21,
    fs=eegf.sample_frequency,
    ylabels=rlabels,
    yscale=2.0,
)

# %% [markdown]
# Ok. That looks pretty good
# - next need to clean up montage viewing code
# - run across all all tuh seizure versions

# %%
