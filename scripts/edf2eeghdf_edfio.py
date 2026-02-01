# -*- coding: utf-8 -*-
"""This will convert an edf file into one encoded in the eeghdf format
it assumes the incoming file has a extension like .edf or .edf+ or .bdf
and produces an output file with the extension .eeg.h5

If the signal labels match the typical ones at Stanford and LPCH, then they are normalized to a standard which looks nicer. Otherwise they are left unchanged.

This will not handle all sorts of EDF files - specifically it will not handle ones
with different sampling rates in different channels

If it is discontinuous, it will be saved as if it was continuous !!!

This version uses the edfio library instead of edflib.

- Added -d or --directory argument which will search a directory recursively for all edf/bdf files

- Added --force-convert / -f argument - A new command line flag that forces conversion even if the output file exists

- Added existence check for directory mode - When processing a directory of EDF files, each output file is checked before conversion. If it exists (and --force-convert is not set), it prints a
  skip message.
-  Added existence check for single file mode - Same logic for when converting a single file.

  The skip message format is:
  Skipping (output already exists): <input_file> -> <output_file>

  Usage examples:
  - python edf2eeghdf_edfio.py -d /path/to/edfs - Skips files that already have converted output
  - python edf2eeghdf_edfio.py -d /path/to/edfs --force-convert - Converts all files, overwriting existing outputs

"""

from __future__ import (
    division,
    absolute_import,
    print_function,
)  # py2.6  with_statement

import sys
import pprint
import h5py
import numpy as np
import os.path

# date related stuff
import datetime
import dateutil
import dateutil.tz
import dateutil.parser
import arrow


# compatibility
import future
from future.utils import iteritems
from builtins import range  # range and switch xrange -> range

# from past.builtins import xrange # later, move to from builtins import


import edfio  # Using edfio library instead of edflib
import eeghdf


# debug = print
def debug(*args):
    pass


DEFAULT_EXT = ".eeg.h5"  # default ending for files in the format defined by eeghdf
DEFAULT_BIRTHDATE = datetime.date(1900, 1, 1)

# really need to check the original data type and then save as that datatype along with the necessary conversion factors
# so can convert voltages on own

# try with float32 instead?

# LPCH often uses these labels for electrodes

LPCH_COMMON_1020_LABELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
    "E",
    "PG1",
    "PG2",
    "A1",
    "A2",
    "T1",
    "T2",
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "EEG Mark1",
    "EEG Mark2",
    "Events/Markers",
]

# common 10-20 extended clinical (T1/T2 instead of FT9/FT10)
# will need to specify these as bytes I suppose (or is this ok in utf-8 given the ascii basis)
# keys should be all one case (say upper)
lpch2edf_fixed_len_labels = dict(
    FP1="EEG Fp1         ",
    F7="EEG F7          ",
    T3="EEG T3          ",
    T5="EEG T5          ",
    O1="EEG O1          ",
    F3="EEG F3          ",
    C3="EEG C3          ",
    P3="EEG P3          ",
    FP2="EEG Fp2         ",
    F8="EEG F8          ",
    T4="EEG T4          ",
    T6="EEG T6          ",
    O2="EEG O2          ",
    F4="EEG F4          ",
    C4="EEG C4          ",
    P4="EEG P4          ",
    CZ="EEG Cz          ",
    FZ="EEG Fz          ",
    PZ="EEG Pz          ",
    T1="EEG FT9         ",  # maybe I should map this to FT9/T1
    T2="EEG FT10        ",  # maybe I should map this to FT10/T2
    A1="EEG A1          ",
    A2="EEG A2          ",
    # these are often (?always) EKG at LPCH, note edfspec says use ECG instead
    # of EKG
    X1="ECG X1          ",  # is this invariant? usually referenced to A1
    # this is sometimes ECG but not usually (depends on how squirmy)
    X2="X2              ",
    PG1="EEG Pg1         ",
    PG2="EEG Pg2         ",
    # now the uncommon ones
    NZ="EEG Nz          ",
    FPZ="EEG Fpz         ",
    AF7="EEG AF7         ",
    AF8="EEG AF8         ",
    AF3="EEG AF3         ",
    AFZ="EEG AFz         ",
    AF4="EEG AF4         ",
    F9="EEG F9          ",
    # F7
    F5="EEG F5          ",
    # F3 ='EEG F3          ',
    F1="EEG F1          ",
    # Fz
    F2="EEG F2          ",
    # F4
    F6="EEG F6          ",
    # F8
    F10="EEG F10         ",
    FT9="EEG FT9         ",
    FT7="EEG FT7         ",
    FC5="EEG FC5         ",
    FC3="EEG FC3         ",
    FC1="EEG FC1         ",
    FCZ="EEG FCz         ",
    FC2="EEG FC2         ",
    FC4="EEG FC4         ",
    FC6="EEG FC6         ",
    FT8="EEG FT8         ",
    FT10="EEG FT10        ",
    T9="EEG T9          ",
    T7="EEG T7          ",
    C5="EEG C5          ",
    # C3 above
    C1="EEG C1          ",
    # Cz above
    C2="EEG C2          ",
    # C4 ='EEG C4          ',
    C6="EEG C6          ",
    T8="EEG T8          ",
    T10="EEG T10         ",
    # A2
    # T3
    # T4
    # T5
    # T6
    TP9="EEG TP9         ",
    TP7="EEG TP7         ",
    CP5="EEG CP5         ",
    CP3="EEG CP3         ",
    CP1="EEG CP1         ",
    CPZ="EEG CPz         ",
    CP2="EEG CP2         ",
    CP4="EEG CP4         ",
    CP6="EEG CP6         ",
    TP8="EEG TP8         ",
    TP10="EEG TP10        ",
    P9="EEG P9          ",
    P7="EEG P7          ",
    P5="EEG P5          ",
    # P3
    P1="EEG P1          ",
    # Pz
    P2="EEG P2          ",
    # P4
    P6="EEG P6          ",
    P8="EEG P8          ",
    P10="EEG P10         ",
    PO7="EEG PO7         ",
    PO3="EEG PO3         ",
    POZ="EEG POz         ",
    PO4="EEG PO4         ",
    PO8="EEG PO8         ",
    # O1
    OZ="EEG Oz          ",
    # O2
    IZ="EEG Iz          ",
)


LPCH_TO_STD_LABELS_STRIP = {
    k: v.strip() for k, v in iteritems(lpch2edf_fixed_len_labels)
}

# this is not used currently but shows how the lpch/stanford labels map to
# EDF standard text labels
LPCH_COMMON_1020_LABELS_to_EDF_STANDARD = {
    "Fp1": "Fp1-REF",
    "Fp2": "Fp2-REF",
    "F3": "F3-REF",
    "F4": "F4-REF",
    "C3": "C3-REF",
    "C4": "C4-REF",
    "P3": "P3-REF",
    "P4": "P4-REF",
    "O1": "O1-REF",
    "O2": "O2-REF",
    "F7": "F7-REF",
    "F8": "F8-REF",
    "T3": "T3-REF",
    "T4": "T4-REF",
    "T5": "T5-REF",
    "T6": "T6-REF",
    "Fz": "Fz-REF",
    "Cz": "Cz-REF",
    "Pz": "Pz-REF",
    "PG1": "PG1-REF",
    "PG2": "PG2-REF",
    "A1": "A1-REF",
    "A2": "A2-REF",
    "T1": "T1-REF",
    "T2": "T2-REF",
    "X1": "X1-REF",
    "X2": "X2-REF",
    "X3": "X3-REF",
    "X4": "X4-REF",
    "X5": "X5-REF",
    "X6": "X6-REF",
    "X7": "X7-REF",
}


def normalize_lpch_signal_label(label):
    """The electrodes at Stanford and LPCH are fixed length strings with lots of spaces in the NK files
    this passes through non-matching labels"""
    uplabel = label.upper()
    if uplabel in LPCH_TO_STD_LABELS_STRIP:
        return LPCH_TO_STD_LABELS_STRIP[uplabel]
    else:
        return label


def edf2h5_float32(fn, outfn="", hdf_dir="", anonymous=False):
    """
    convert an edf file to hdf5 using a straightforward mapping
    convert to real-valued signals store as float32's

    just getting started here
    --- metadata ---
    number_signals
    sample_frequency
    nsamples
    age
    signal_labels

    Post Menstrual Age
    """
    if not outfn:
        base = os.path.basename(fn)
        base, ext = os.path.splitext(base)

        base = base + DEFAULT_EXT
        outfn = os.path.join(hdf_dir, base)
        debug("outfn:", outfn)

    edf = edfio.read_edf(fn)

    nsigs = edf.num_signals
    # again know/assume that this is uniform sampling across signals
    fs = [sig.sampling_frequency for sig in edf.signals]
    fs0 = fs[0]

    if any([fs0 != xx for xx in fs]):
        print("error caught multiple sampling frequencies in edf files!!!")
        sys.exit(0)

    nsamples0 = len(edf.signals[0].data)

    debug("nsigs=%s, fs0=%s, nsamples0=%s" % (nsigs, fs0, nsamples0))

    # create file 'w-' -> fail if exists , w -> truncate if exists
    hdf = h5py.File(outfn, "w")
    # use compression? yes! give it a try
    eegdata = hdf.create_dataset(
        "eeg",
        (nsigs, nsamples0),
        dtype="float32",
        # chunks=(nsigs,fs0),
        chunks=True,
        fletcher32=True,
        # compression='gzip',
        # compression='lzf',
        # maxshape=(256,None)
    )
    # no compression     -> 50 MiB     can view eegdata in vitables
    # compression='gzip' -> 27 MiB    slower
    # compression='lzf'  -> 35 MiB
    # compression='lzf' maxshape=(256,None) -> 36MiB
    # szip is unavailable
    patient = hdf.create_group("patient")

    # add meta data
    hdf.attrs["number_signals"] = nsigs
    hdf.attrs["sample_frequency"] = fs0
    hdf.attrs["nsamples0"] = nsamples0

    # Patient info from edfio
    patient.attrs["gender_b"] = (
        edf.patient.sex if edf.patient and edf.patient.sex else ""
    )
    patient.attrs["patientname"] = (
        edf.patient.name if edf.patient and edf.patient.name else ""
    )  # PHI

    debug(
        "birthdate: %s" % (edf.patient.birthdate if edf.patient else None),
        type(edf.patient.birthdate if edf.patient else None),
    )
    # this is already a date object in edfio
    if not (edf.patient and edf.patient.birthdate):
        debug("no birthday in this file")
        birthdate = None
    else:
        birthdate = edf.patient.birthdate
        debug("birthdate (date object):", birthdate)

    # Combine startdate and starttime to get start_date_time
    start_date_time = datetime.datetime.combine(edf.startdate, edf.starttime)
    debug(start_date_time)
    if start_date_time and birthdate:
        age = start_date_time - datetime.datetime.combine(birthdate, datetime.time())
        debug("age:", age)
    else:
        age = None

    if age:
        patient.attrs["post_natal_age_days"] = age.days
    else:
        patient.attrs["post_natal_age_days"] = -1

    # now start storing the lists of things: labels, units...
    # nsigs = len(label_list)
    # variable ascii string (or b'' type)
    str_dt = h5py.special_dtype(vlen=str)
    label_ds = hdf.create_dataset("signal_labels", (nsigs,), dtype=str_dt)
    units_ds = hdf.create_dataset("signal_units", (nsigs,), dtype=str_dt)
    labels = []
    units = list()
    # signal_nsamples = []
    for sig in edf.signals:
        labels.append(sig.label)
        units.append(sig.physical_dimension)
        # self.signal_nsamples.append(self.cedf.samples_in_file(ii))
        # self.samplefreqs.append(self.cedf.samplefrequency(ii))
    # eegdata.signal_labels = labels
    # labels are fixed length strings
    labels_strip = [ss.strip() for ss in labels]
    label_ds[:] = labels_strip
    units_ds[:] = units
    # should be more and a switch for anonymous or not

    # need to change this to

    nchunks = int(nsamples0 // fs0)
    samples_per_chunk = int(fs0)
    buf = np.zeros((nsigs, samples_per_chunk), dtype="float64")  # buffer is float64

    debug("nchunks: ", nchunks, "samples_per_chunk:", samples_per_chunk)

    bookmark = 0  # mark where we are in samples
    for ii in range(nchunks):
        for jj in range(nsigs):
            # With edfio, we can directly slice the data array
            buf[jj] = edf.signals[jj].data[bookmark : bookmark + samples_per_chunk]
        # conversion from float64 to float32
        eegdata[:, bookmark : bookmark + samples_per_chunk] = buf
        # bookmark should be ii*fs0
        bookmark += samples_per_chunk
    left_over_samples = nsamples0 - nchunks * samples_per_chunk
    debug("left_over_samples:", left_over_samples)

    if left_over_samples > 0:
        for jj in range(nsigs):
            buf[jj] = edf.signals[jj].data[bookmark : bookmark + left_over_samples]
        eegdata[:, bookmark : bookmark + left_over_samples] = buf[
            :, 0:left_over_samples
        ]
    hdf.close()


def edf_block_iter_generator(edf, nsamples, samples_per_chunk, dtype="int32"):
    """
    factory to produce generators for iterating through an edf file and filling
    up an array from the edf with the signal data starting at 0. You choose the
    number of @samples_per_chunk, and number of samples to do in total
    @nsamples as well as the dtype. 'int16' is reasonable as well 'int32' will
    handle everything though


    it yields -> (numpy_buffer, mark, num)
        numpy_buffer,
        mark, which is where in the file in total currently reading from
        num   -- which is the number of samples in the buffer (per signal) to transfer
    """

    nchan = edf.num_signals

    # 'int32' will work for int16 as well
    buf = np.zeros((nchan, samples_per_chunk), dtype=dtype)

    nchunks = nsamples // samples_per_chunk
    left_over_samples = nsamples - nchunks * samples_per_chunk

    mark = 0
    for ii in range(nchunks):
        for cc in range(nchan):
            # With edfio, access digital values directly
            buf[cc] = edf.signals[cc].digital[mark : mark + samples_per_chunk]

        yield (buf, mark, samples_per_chunk)
        mark += samples_per_chunk
        # debug('mark:', mark)
    # left overs
    if left_over_samples > 0:
        for cc in range(nchan):
            buf[cc, :left_over_samples] = edf.signals[cc].digital[
                mark : mark + left_over_samples
            ]

        yield (buf[:, 0:left_over_samples], mark, left_over_samples)


def dig2phys(eeghdf, start, end, chstart, chend):
    # edfhdr->edfparam[i].bitvalue = (edfhdr->edfparam[i].phys_max - edfhdr->edfparam[i].phys_min) / (edfhdr->edfparam[i].dig_max - edfhdr->edfparam[i].dig_min);
    # edfhdr->edfparam[i].offset = edfhdr->edfparam[i].phys_max /
    # edfhdr->edfparam[i].bitvalue - edfhdr->edfparam[i].dig_max;
    dmins = eeghdf["signal_digital_mins"][:]
    dmaxs = eeghdf["signal_digital_maxs"][:]
    phys_maxs = eeghdf["signal_physical_maxs"][:]
    phys_mins = eeghdf["signal_physical_mins"][:]
    debug("dmaxs:", repr(dmaxs))
    debug("dmins:", repr(dmins))
    debug("dmaxs[:] - dmins[:]", dmaxs - dmins)
    debug("phys_maxs", phys_maxs)
    debug("phys_mins", phys_mins)
    bitvalues = (phys_maxs - phys_mins) / (dmaxs - dmins)
    offsets = phys_maxs / bitvalues - dmaxs
    debug("bitvalues, offsets:", bitvalues, offsets)
    debug("now change their shape to column vectors")
    for arr in (bitvalues, offsets):
        if len(arr.shape) != 1:
            debug("logical error %s shape is unexpected" % arr.shape)
            raise Exception
        s = arr.shape
        arr.shape = (s[0], 1)
    debug("bitvalues, offsets:", bitvalues, offsets)
    # buf[i] = phys_bitvalue * (phys_offset + (double)var.two_signed[0]);
    dig_signal = eeghdf["signals"][chstart:chend, start:end]
    # signal = bitvalues[chstart:chend] *(dig_signal[chstart:chend,:] + offsets[chstart:chend])
    phys_signals = (dig_signal[:, start:end] + offsets) * bitvalues
    # return signal, bitvalues, offsets
    return phys_signals


# TODO: create edf -> hdf version 1000
# hdf -> edf for hdf version 1000
# tests to verify that round trip is lossless
# [] writing encoding of MRN
# [] and entry of mapped pt_code into database coe

# Plan
# v = ValidateTrackHeader(header=h)
# if v.is_valid():
#     process(v.cleaned_data)
# else:
#    mark_as_invalid(h)


def first(mapping):
    if mapping:
        return mapping[0]
    else:
        return mapping  # say mapping = [] or None


def create_simple_anonymous_header(header):
    hdr = header.copy()
    debug(f"header: {hdr}")
    hdr["patient_name"] = "anonymous"
    if hdr["patientcode"]:
        hdr["patientcode"] = "00000000"
    dob = hdr["birthdate_date"]
    if not hdr["birthdate_date"]:  # 0-len string or None
        age_time_offset = datetime.timedelta(seconds=0)
    else:
        age_time_offset = hdr["birthdate_date"] - DEFAULT_BIRTHDATE
    hdr["birthdate_date"] = DEFAULT_BIRTHDATE

    hdr["file_name"]
    if hdr["start_datetime"]:
        hdr["start_datetime"] = hdr["start_datetime"] - age_time_offset

    if hdr["startdate_date"]:
        hdr["startdate_date"] = hdr["start_datetime"].date()

    hdr["file_duration_seconds"]

    hdr["admincode"] = ""
    hdr["technician"] = ""
    hdr["patient_additional"] = ""
    debug(f"anonymized hdr:")
    debug(pprint.pformat(hdr))
    return hdr


def find_blocks(arr):
    blocks = []
    debug("total arr:", arr)
    dfs = np.diff(arr)
    dfs_ind = np.where(dfs != 0.0)[0]
    last_ind = 0
    for dd in dfs_ind + 1:
        debug("block:", arr[last_ind:dd])
        blocks.append((last_ind, dd))
        last_ind = dd
    debug("last block:", arr[last_ind:])
    blocks.append((last_ind, len(arr)))
    return blocks


def find_blocks2(arr):
    blocks = []
    N = len(arr)
    debug("total arr:", arr)
    last_ind = 0
    last_val = arr[0]
    for ii in range(1, N):
        if last_val == arr[ii]:
            pass
        else:
            blocks.append((last_ind, ii))
            last_ind = ii
            last_val = arr[ii]
    blocks.append((last_ind, N))
    return blocks


def test_find_blocks1():
    s = [250.0, 250.0, 250.0, 1.0, 1.0, 1000.0, 1000.0]
    blocks = find_blocks(s)
    debug("blocks:")
    debug(blocks)


def test_find_blocks2():
    s = [250.0, 250.0, 250.0, 1.0, 1.0, 1000.0, 1000.0]
    blocks = find_blocks2(s)
    debug("blocks:")
    debug(blocks)


def test_find_blocks2_2():
    s = [100, 100, 100, 100, 100, 100, 100, 100]
    blocks = find_blocks2(s)
    debug("blocks:")
    debug(blocks)


def edf2hdf(fn, outfn="", hdf_dir="", anonymize=False):
    """
    convert an edf file to hdf5 using fairly straightforward mapping
    return True if successful

    by default (if outfn and hdf_dir are not set)
       the output is put in the same directory as the input file
    you can also specify the output file (full path) by setting outfn directly
    or simple specify a different target directory by specifying @hdf_dir as a directory path

    @database_source_label tells us which database it came from LPCH_NK or STANFORD_NK
       this is important!
    """

    if not outfn:
        parentdir = os.path.dirname(fn)
        base = os.path.basename(fn)
        base, ext = os.path.splitext(base)

        base = base + DEFAULT_EXT
        if hdf_dir:
            outfn = os.path.join(hdf_dir, base)
        else:
            outfn = os.path.join(parentdir, base)
            # debug('outfn:', outfn)
        # all the data point related stuff

    # Read the EDF file with edfio
    edf = edfio.read_edf(fn)

    # read all EDF+ header information in just the way I want it
    # Convert edfio's patient and recording objects to the header dict format
    patient = edf.patient
    recording = edf.recording

    # Determine filetype based on reserved field
    if edf.reserved == "EDF+C":
        filetype = 1  # EDF+C
    elif edf.reserved == "EDF+D":
        filetype = 2  # EDF+D
    else:
        filetype = 0  # EDF

    # Handle potentially anonymized dates
    try:
        birthdate = patient.birthdate if patient else None
    except edfio.AnonymizedDateError:
        birthdate = None

    try:
        startdate = edf.startdate
    except edfio.AnonymizedDateError:
        startdate = datetime.date(1985, 1, 1)  # Default anonymized date

    header = {
        "file_name": os.path.basename(fn),
        "filetype": filetype,
        "patient_name": patient.name if patient and patient.name else "X",
        "patientcode": patient.code if patient and patient.code else "X",
        "studyadmincode": recording.hospital_administration_code
        if recording and recording.hospital_administration_code
        else "X",
        "gender": patient.sex if patient and patient.sex else "X",
        "signals_in_file": edf.num_signals,
        "datarecords_in_file": edf.num_data_records,
        "file_duration_100ns": int(
            edf.duration * 10_000_000
        ),  # Convert seconds to 100ns units
        "file_duration_seconds": edf.duration,
        "startdate_date": startdate,
        "start_datetime": datetime.datetime.combine(startdate, edf.starttime),
        "starttime_subsecond_offset": edf.starttime.microsecond / 1_000_000
        if edf.starttime.microsecond
        else 0,
        "birthdate_date": birthdate,
        "patient_additional": patient.additional
        if patient and patient.additional
        else "",
        "admincode": recording.hospital_administration_code
        if recording and recording.hospital_administration_code
        else "X",
        "technician": recording.investigator_technician_code
        if recording and recording.investigator_technician_code
        else "X",
        "equipment": recording.equipment_code
        if recording and recording.equipment_code
        else "X",
        "recording_additional": recording.additional
        if recording and recording.additional
        else "",
        "datarecord_duration_100ns": int(
            edf.data_record_duration * 10_000_000
        ),  # Convert seconds to 100ns units
    }
    # debug
    debug("original header")
    debug(pprint.pformat(header))

    #  use arrow
    start_datetime = header["start_datetime"]

    duration = datetime.timedelta(seconds=header["file_duration_seconds"])

    # derived information
    birthdate = header["birthdate_date"]
    if birthdate:
        age = arrow.get(start_datetime) - arrow.get(header["birthdate_date"])

        debug("predicted age: %s" % age)
        # total_seconds() returns a float
        debug("predicted age (seconds): %s" % age.total_seconds())
    else:
        age = datetime.timedelta(seconds=0)

    if anonymize:
        anonymous_header = create_simple_anonymous_header(header)
        header = anonymous_header

    header["end_datetime"] = header["start_datetime"] + duration

    ############# signal array information ##################

    # signal block related stuff
    nsigs = edf.num_signals

    # again know/assume that this is uniform sampling across signals
    fs0 = edf.signals[0].sampling_frequency
    signal_frequency_array = np.array([sig.sampling_frequency for sig in edf.signals])
    dfs = np.diff(signal_frequency_array)
    dfs_ind = np.where(dfs != 0.0)
    dfs_ind = dfs_ind[0]
    last_ind = 0
    for dd in dfs_ind + 1:
        debug("block:", signal_frequency_array[last_ind:dd])
        last_ind = dd
    debug("last block:", signal_frequency_array[last_ind:])

    debug("where does sampling rate change?", np.where(dfs != 0.0))
    debug("elements:", signal_frequency_array[np.where(dfs != 0.0)])
    debug("signal_frequency_array::\n", repr(signal_frequency_array))
    debug("len(signal_frequency_array):", len(signal_frequency_array))

    assert all(signal_frequency_array[:-3] == fs0)

    nsamples0 = len(edf.signals[0].data)  # samples per channel
    debug("nsigs=%s, fs0=%s, nsamples0=%s\n" % (nsigs, fs0, nsamples0))

    num_samples_per_signal = np.array([len(sig.data) for sig in edf.signals])
    debug("num_samples_per_signal::\n", repr(num_samples_per_signal), "\n")

    # assert all(num_samples_per_signal == nsamples0)

    file_duration_sec = edf.duration
    # debug("file_duration_sec", repr(file_duration_sec))

    # Note that all annotations except the top row must also specify a duration.

    # In edfio, annotations are EdfAnnotation namedtuples with (onset, duration, text)
    # onset is in seconds (float), duration is optional (can be None or "")
    # Convert to format expected by eeghdf writer: [onset_100ns, duration_bytes, text_bytes]
    annotations_b = []
    for ann in edf.annotations:
        # Convert to 100ns units
        onset_100ns = int(ann.onset * 10_000_000)
        # Duration as bytes string
        duration_bytes = str(ann.duration).encode("utf-8") if ann.duration else b""
        # Text as bytes
        text_bytes = ann.text.encode("utf-8") if ann.text else b""
        annotations_b.append([onset_100ns, duration_bytes, text_bytes])

    # debug("annotations_b::\n")
    # pprint.pprint(annotations_b)  # get list of annotations

    signal_text_labels = [sig.label for sig in edf.signals]
    debug("signal_text_labels::\n")
    debug(pprint.pformat(signal_text_labels))
    debug("normalized text labels::\n")
    signal_text_labels_lpch_normalized = [
        normalize_lpch_signal_label(label) for label in signal_text_labels
    ]
    debug(pprint.pformat(signal_text_labels_lpch_normalized))

    # ef.recording_additional

    # debug()
    signal_digital_mins = np.array([sig.digital_min for sig in edf.signals])
    signal_digital_total_min = min(signal_digital_mins)

    debug("digital mins:", repr(signal_digital_mins))
    debug("digital total min:", repr(signal_digital_total_min))

    signal_digital_maxs = np.array([sig.digital_max for sig in edf.signals])
    signal_digital_total_max = max(signal_digital_maxs)

    debug("digital maxs:", repr(signal_digital_maxs))
    # debug("digital total max:", repr(signal_digital_total_max))

    signal_physical_dims = [sig.physical_dimension for sig in edf.signals]
    # debug('signal_physical_dims::\n')
    # pprint.pformat(signal_physical_dims)
    # debug()

    signal_physical_maxs = np.array([sig.physical_max for sig in edf.signals])

    # debug('signal_physical_maxs::\n', repr(signal_physical_maxs))

    signal_physical_mins = np.array([sig.physical_min for sig in edf.signals])

    # debug('signal_physical_mins::\n', repr(signal_physical_mins))

    # this don't seem to be used much so I will put at end
    signal_prefilters = [sig.prefiltering.strip() for sig in edf.signals]
    # debug('signal_prefilters::\n')
    # pprint.pformat(signal_prefilters)
    # debug()
    signal_transducers = [sig.transducer_type.strip() for sig in edf.signals]
    # debug('signal_transducers::\n')
    # pprint.pformat(signal_transducers)

    with eeghdf.EEGHDFWriter(outfn, "w") as eegf:
        eegf.write_patient_info(
            patient_name=header["patient_name"],
            patientcode=header["patientcode"],
            gender=header["gender"],
            birthdate_isostring=str(header["birthdate_date"])
            if header["birthdate_date"]
            else "",
            # gestational_age_at_birth_days
            # born_premature
            patient_additional=header["patient_additional"],
        )

        signal_text_labels_lpch_normalized = [
            normalize_lpch_signal_label(label) for label in signal_text_labels
        ]

        rec = eegf.create_record_block(
            record_duration_seconds=header["file_duration_seconds"],
            start_isodatetime=str(header["start_datetime"]),
            end_isodatetime=str(header["end_datetime"]),
            number_channels=header["signals_in_file"],
            num_samples_per_channel=nsamples0,
            sample_frequency=fs0,
            signal_labels=signal_text_labels_lpch_normalized,
            signal_physical_mins=signal_physical_mins,
            signal_physical_maxs=signal_physical_maxs,
            signal_digital_mins=signal_digital_mins,
            signal_digital_maxs=signal_digital_maxs,
            physical_dimensions=signal_physical_dims,
            patient_age_days=age.total_seconds() / 86400.0,
            signal_prefilters=signal_prefilters,
            signal_transducers=signal_transducers,
            technician=header["technician"],
            studyadmincode=header["studyadmincode"],
        )

        eegf.write_annotations_b(
            annotations_b
        )  # may be should be called record annotations

        # Get samples_per_data_record for first signal
        # In edfio, this is samples_per_data_record property
        samples_per_datarecord = edf.signals[0].samples_per_data_record

        edfblock_itr = edf_block_iter_generator(
            edf,
            nsamples0,
            100
            * samples_per_datarecord
            * header[
                "signals_in_file"
            ],  # samples_per_chunk roughly 100 datarecords at a time
            dtype="int32",
        )

        signals = eegf.stream_dig_signal_to_record_block(rec, edfblock_itr)

    return True


def test_edf2hdf_info():
    # on chris's macbook
    EDF_DIR = r"/Users/clee/code/eegml/nk_database_proj/private/lpch_edfs"
    fn = os.path.join(EDF_DIR, "XA2731AX_1-1+.edf")
    edf2hdf(fn)


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="input EDF file (required if --directory not specified)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="directory containing EDF files to convert (if specified, filename is ignored)",
    )
    parser.add_argument(
        "--output-file", "-o", type=str, help="name of output file", default=""
    )
    parser.add_argument(
        "--anonymize",
        "-a",
        action="store_true",
        default=False,
        help="add flag to do simple anonymization of file patient information",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        default=False,
        help="print what would be done without actually converting files",
    )
    parser.add_argument(
        "--force-convert",
        "-f",
        action="store_true",
        default=False,
        help="force conversion even if output file already exists",
    )
    args = parser.parse_args()
    debug(args)

    if args.directory:
        # Process all EDF files in the directory recursively
        edf_files = glob.glob(
            os.path.join(args.directory, "**", "*.edf"), recursive=True
        )
        edf_files += glob.glob(
            os.path.join(args.directory, "**", "*.EDF"), recursive=True
        )
        edf_files += glob.glob(
            os.path.join(args.directory, "**", "*.bdf"), recursive=True
        )
        edf_files += glob.glob(
            os.path.join(args.directory, "**", "*.BDF"), recursive=True
        )
        edf_files = list(set(edf_files))  # Remove duplicates

        if not edf_files:
            print(f"No EDF/BDF files found in {args.directory}")
            sys.exit(1)

        print(f"Found {len(edf_files)} EDF/BDF files to convert")
        for edf_file in sorted(edf_files):
            outfn = os.path.splitext(edf_file)[0] + DEFAULT_EXT
            if args.dry_run:
                print(f"Would convert: {edf_file} -> {outfn}")
            elif os.path.exists(outfn) and not args.force_convert:
                print(f"Skipping (output already exists): {edf_file} -> {outfn}")
            else:
                print(f"Converting: {edf_file}")
                try:
                    edf2hdf(edf_file, anonymize=args.anonymize)
                except Exception as e:
                    print(f"Error converting {edf_file}: {e}")
    else:  # single file mode
        if args.filename is None:
            parser.error("filename is required when --directory is not specified")
        outfn = (
            args.output_file
            if args.output_file
            else os.path.splitext(args.filename)[0] + DEFAULT_EXT
        )
        if args.dry_run:
            print(f"Would convert: {args.filename} -> {outfn}")
        elif os.path.exists(outfn) and not args.force_convert:
            print(f"Skipping (output already exists): {args.filename} -> {outfn}")
        else:
            edf2hdf(args.filename, outfn=args.output_file, anonymize=args.anonymize)
