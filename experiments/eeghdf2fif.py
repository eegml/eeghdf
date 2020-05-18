# coding: utf-8

import mne
import eeghdf
import eegvis.stacklineplot as stackplot
import matplotlib
import matplotlib.pyplot as plt

#%% # data path
DATAPATH='/mnt/home2/clee/code/eegml/eeghdf/data'
#%%
# check versions

print('matplotlib.__version__')
#%%
plt.rcParams['figure.figsize'] = (24,9)

# start to make this into a function 


def ehdf2mne(hf):
    """@hf is an eeghdf Eeghdf object opened on a file
    from the stanford EEG corpus """
    
    # start to make this into a function 
    # find useful_channels

    useful_channels = []
    useful_channel_labels = []

    for ii, label in enumerate(hf.electrode_labels):
        if label.find('Mark') >= 0:
            continue
        if label.find('EEG') >= 0: # we use standard names
            useful_channels.append(ii)
            useful_channel_labels.append(label)

    # add ECG if there
    for ii, label in enumerate(hf.electrode_labels):
        if label.find('ECG') >= 0:
            useful_channels.append(ii)
            useful_channel_labels.append(label)

    print(list(zip(useful_channels, useful_channel_labels)))

    num_uchans = len(useful_channels)


    def label2type(name):
        """lots of assumptions to use this as name is already limited"""
        try:
            if name[:6] == 'EEG Pg':
                return 'eog'
        except:
            pass

        if name[:3] == 'ECG':
            return 'ecg'
        if name[:3] == 'EEG':
            return 'eeg'
        return 'misc'



    channel_types = [label2type(label) for label in useful_channel_labels]
    print(channel_types)

    # now get rid of the prefixes
    uchan_names = [ss.split()[1] if ss[:3] == 'EEG' else ss 
                   for ss in useful_channel_labels]

    print('final view before sending to info')
    for ii, name in enumerate(uchan_names):
        print(ii, name, channel_types[ii])

    # finally remove the prefix 'EEG' from the label names


    # info - mne.create_info(unum_uchans, hf.sample_frequency)
    info = mne.create_info(uchan_names, hf.sample_frequency, 
                           channel_types, montage='standard_1020')
    print(info)


    #montage = 'standard_1010' # might work

    # start on the data

    data = hf.phys_signals[useful_channels, :]

    # MNE wants EEG and ECG in Volts
    for jj, ii in enumerate(useful_channels):
        unit = hf.physical_dimensions[ii]
        if unit == 'uV':
            data[jj, :] = data[jj, :]/1000000
        if unit == 'mV':
            data[jj, :] = data[jj, :]/1000

    print(data.shape)
    # TODO: transfer recording and patient details. API ref 
    # url: https://martinos.org/mne/dev/generated/mne.Info.html#mne.Info
    # TODO: next need to figure out how to add the events/annotations
    info['custom_ref_applied'] = True # for SEC this is true

    # events are a list of dict events list of dict:

    # channels : list of int
    # Channel indices for the events.
    # event dict:
    # 'channels' : list|ndarray of int|int32 # channel indices for the evnets
    # 'list' : ndarray, shape (n_events * 3,)
    #          triplets as number of samples, before, after.
    # I will need to see an exmaple of this
    # info['highpass'], info['lowpass']

    info['line_freq'] = 60.0

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

    # work on adding annotations 
    starts_sec = [1e-7 * t100ns for t100ns in hf._annotation_start100ns]
    mne_annot = mne.Annotations(onset=starts_sec, duration=hf._annotation_durations_sec,
                                description=hf._annotation_text)

    customraw = mne.io.RawArray(data, info)
    customraw.set_annotations(mne_annot)    


    return customraw, info, useful_channels



if __name__ == '__main__':
#%%
    hf = eeghdf.Eeghdf(DATAPATH + '/absence_epilepsy.eeghdf')
    channel_number, num_samples = hf.phys_signals.shape
    print('original shape:', (channel_number, num_samples) )
    print('number channels:', hf.number_channels)

#%% 
    # the eeghdf annotations currently only have useful info about start
    # time and description. The duration field seems to always be cut
    # off but that may not always be the case
    # oaccording to the edf spec 
    # the durations are stored as text (ascii) numbers in seconds
    # if they are null I will consider duration = 0

    # hf.hdf['record-0']['edf_annotations']

    raw, info, chans = ehdf2mne(hf)

    raw.plot()
