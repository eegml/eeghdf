# -*- coding: utf-8 -*-
"""
some basic conversion utilities
I'm not sure how useful these will be as every source may require some variation in conversion 
"""

# convert eeg-hdf storage to mne raw object
import mne


# TODO !!!:
# def hdf2edf(hf) # so can reverse process


def sechdf1020_to_mne(hf):
    """@hf is an eeghdf Eeghdf object opened on a file
    from the stanford EEG corpus for a scalp EEG using
    the 10-20 system. It won't be correct montage for iEEGs"""

    # start to make this into a function
    # find useful_channels

    useful_channels = []
    useful_channel_labels = []

    for ii, label in enumerate(hf.electrode_labels):
        if label.find("Mark") >= 0:
            continue
        if label.find("EEG") >= 0:  # we use standard names
            useful_channels.append(ii)
            useful_channel_labels.append(label)

    # add ECG if there
    for ii, label in enumerate(hf.electrode_labels):
        if label.find("ECG") >= 0:
            useful_channels.append(ii)
            useful_channel_labels.append(label)

    print(list(zip(useful_channels, useful_channel_labels)))

    num_uchans = len(useful_channels)

    def label2type(name):
        """lots of assumptions to use this as name is already limited"""
        try:
            if name[:6] == "EEG Pg":
                return "eog"
        except:
            pass

        if name[:3] == "ECG":
            return "ecg"
        if name[:3] == "EEG":
            return "eeg"
        return "misc"

    channel_types = [label2type(label) for label in useful_channel_labels]
    print(channel_types)

    # now get rid of the prefixes
    uchan_names = [
        ss.split()[1] if ss[:3] == "EEG" else ss for ss in useful_channel_labels
    ]

    print("final view before sending to info")
    for ii, name in enumerate(uchan_names):
        print(ii, name, channel_types[ii])

    # finally remove the prefix 'EEG' from the label names

    # info - mne.create_info(unum_uchans, hf.sample_frequency)
    info = mne.create_info(
        uchan_names, hf.sample_frequency, channel_types, # montage="standard_1020"
    )
    ten_twenty_montage = mne.channels.make_standard_montage("standard_1020")
    print(info)

    # montage = 'standard_1010' # might work

    # start on the data

    data = hf.phys_signals[useful_channels, :]

    # MNE wants EEG and ECG in Volts
    for jj, ii in enumerate(useful_channels):
        unit = hf.physical_dimensions[ii]
        if unit == "uV":
            data[jj, :] = data[jj, :] / 1000000
        if unit == "mV":
            data[jj, :] = data[jj, :] / 1000

    print(data.shape)
    # TODO: transfer recording and patient details. API ref
    # url: https://martinos.org/mne/dev/generated/mne.Info.html#mne.Info
    # TODO: next need to figure out how to add the events/annotations
    # info["custom_ref_applied"] = True  # for SEC this is true

    # events are a list of dict events list of dict:

    # channels : list of int
    # Channel indices for the events.
    # event dict:
    # 'channels' : list|ndarray of int|int32 # channel indices for the evnets
    # 'list' : ndarray, shape (n_events * 3,)
    #          triplets as number of samples, before, after.
    # I will need to see an exmaple of this
    # info['highpass'], info['lowpass']

    info["line_freq"] = 60.0

    # info['subject_info'] = <dict>
    # subject_info dict:

    # id : int
    # Integer subject identifier.

    # his_id : str
    # String subject identifier.

    # last_name : str
    # Last name.

    # first_name : str
    # First name.

    # middle_name : str
    # Middle name.

    # birthday : tuple of int
    # Birthday in (year, month, day) format.

    # sex : int
    # Subject sex (0=unknown, 1=male, 2=female).

    customraw = mne.io.RawArray(data, info)
    customraw.set_montage(ten_twenty_montage)
    return customraw, info, useful_channels


def hdf2mne(hf):
    """convert generic eeg hdf5 object to mne raw object

    @hf is a generic eeghdf Eeghdf object opened on a file

    if you are using the stanford eeg corpus (SEC) then 
    you should use sechdf1020_to_mne as this can assume more things
    can put more info in the raw object.

    In many cases for example the recording was done using 10-20 system
    electrodes + EOG and ECG monitors and this does not respect that

    This just dumps all the data in as misc channels and expects the user to
    figure out which channels types are which

    you can use customraw.set_montage(<montage>) if you know the montage used
    """

    # start to make this into a function
    # find useful_channels

    useful_channels = []
    useful_channel_labels = []

    for ii, label in enumerate(hf.electrode_labels):
        if label.find("Mark") >= 0:
            continue
        if label.find("EEG") >= 0:  # we use standard names
            useful_channels.append(ii)
            useful_channel_labels.append(label)

    # add ECG if there
    for ii, label in enumerate(hf.electrode_labels):
        if label.find("ECG") >= 0:
            useful_channels.append(ii)
            useful_channel_labels.append(label)

    print(list(zip(useful_channels, useful_channel_labels)))

    num_uchans = len(useful_channels)

    def label2type(name):
        """lots of assumptions to use this as name is already limited
        it is more or less based upon the "standard texts" the edf 
        specification"""
        try:
            if (
                name[:6] == "EEG Pg"
            ):  # this is a decision but could figure this out better
                return "eog"
        except:
            pass

        if name[:3] == "ECG":
            return "ecg"
        if name[:3] == "EEG":
            return "eeg"
        if name[:3] == "EMG":
            return "emg"
        if name[:3] == "MEG":
            return "meg"
        if name[:3] == "MCG":
            return "misc"  # 'mcg' ?
        if name[:3] == "EP":
            return "eeg"  # evoked potential
        if name[:3] == "ERG":  # electroretinogram
            return "misc"
        if name[:4] == "Temp":
            return "misc"  #
        if name[:4] == "SaO2":
            return "misc"
        if name[:5] == "Light":
            return "misc"
        if name[:5] == "Sound":
            return "misc"
        if name[:5] == "Event":  # event button
            return "misc"
        return "misc"

    channel_types = [label2type(label) for label in useful_channel_labels]
    print(channel_types)

    # now get rid of the prefixes
    uchan_names = [
        ss.split()[1] if ss[:3] == "EEG" else ss for ss in useful_channel_labels
    ]

    print("final view before sending to info")
    for ii, name in enumerate(uchan_names):
        print(ii, name, channel_types[ii])

    # finally remove the prefix 'EEG' from the label names
    # mne.create_info no longer includes montage argument
    info = mne.create_info(
        uchan_names, hf.sample_frequency, channel_types,
    )

    print(info)

    # montage = 'standard_1010' # might work

    # start on the data

    data = hf.phys_signals[useful_channels, :]

    # MNE wants EEG and ECG in Volts
    for jj, ii in enumerate(useful_channels):
        unit = hf.physical_dimensions[ii]
        if unit == "uV":
            data[jj, :] = data[jj, :] / 1000000
        if unit == "mV":
            data[jj, :] = data[jj, :] / 1000

    print(data.shape)
    # TODO: transfer recording and patient details. API ref
    # url: https://martinos.org/mne/dev/generated/mne.Info.html#mne.Info
    # TODO: next need to figure out how to add the events/annotations
    #info["custom_ref_applied"] = True  # for SEC this is true
    # use inst.set_eeg_reference() instead.
    
    # events are a list of dict events list of dict:

    # channels : list of int
    # Channel indices for the events.
    # event dict:
    # 'channels' : list|ndarray of int|int32 # channel indices for the evnets
    # 'list' : ndarray, shape (n_events * 3,)
    #          triplets as number of samples, before, after.
    # I will need to see an exmaple of this
    # info['highpass'], info['lowpass']

    info["line_freq"] = 60.0

    # info['subject_info'] = <dict>
    # subject_info dict:

    # id : int
    # Integer subject identifier.

    # his_id : str
    # String subject identifier.

    # last_name : str
    # Last name.

    # first_name : str
    # First name.

    # middle_name : str
    # Middle name.

    # birthday : tuple of int
    # Birthday in (year, month, day) format.

    # sex : int
    # Subject sex (0=unknown, 1=male, 2=female).

    customraw = mne.io.RawArray(data, info)

    return customraw, info, useful_channels
