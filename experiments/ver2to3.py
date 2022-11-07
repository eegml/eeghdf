# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3.8.8 ('pyt181')
#     language: python
#     name: python3
# ---

# %%
import eeghdf 
import h5py

# %% [markdown]
# The purpose of this file is to take version 2 eeghdf files and turn them into version3 files
#
# The main thing to do is to create a 
#
# record_groups dataset
#
# DT_REF = h5py.special_dtype(ref=h5py.Reference)

# %%
DT_REF = h5py.special_dtype(ref=h5py.Reference)

# %%
sample_file = '/home/clee/code/eegml/stevenson_neonatalsz_2019/hdf_w_annotations/eeg1.annot.eeg.h5'

hf = h5py.File(sample_file, 'r+')

# %%
keys = list(hf.keys())
keys

# %%
rec_group_keys = [kk for kk in list(hf.keys()) if kk.find('record-') != -1]
rec_group_keys

# %%
keys

# %%
rec_reflist = [hf[kk].ref for kk in rec_group_keys]
rec_reflist

# %%
num_recs = len(rec_reflist)
ref_dset = hf.create_dataset("recording_refs", (num_recs,),dtype=DT_REF)

# %%
ref_dset[:] = rec_reflist[:]

# %%
ref_dset

# %%
ref_dset[:]

# %%
