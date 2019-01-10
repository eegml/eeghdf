# coding: utf-8

import mne
import eeghdf
import eegvis.stacklineplot as stackplot
import matplotlib
import matplotlib.pyplot as plt

#%% # data path
DATAPATH='/mnt/home2/clee/code/eegml/eeg-hdfstorage/data'
#%%
# check versions

print('matplotlib.__version__')
#%%
plt.rcParams['figure.figsize'] = (24,9)

#%%
hf = eeghdf.Eeghdf(DATAPATH + '/absence_epilepsy.eeghdf')
channel_number, num_samples = hf.phys_signals.shape
print('original shape:', (channel_number, num_samples) )
print('number channles:', hf.number_channels)
#%%
# find useful_channels

useful_channels = []
useful_channel_labels = []
for ii,label in enumerate(hf.electrode_labels):
    if label.find('Mark') >= 0:
        continue
    
    if label.find('EEG') >= 0: # we use standard names
        useful_channels.append(ii)
        useful_channel_labels.append(label)
# add ECG if there
for ii,label in enumerate(hf.electrode_labels):
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
uchan_names = [ss.split()[1] if ss[:3]=='EEG' else ss for ss in useful_channel_labels]

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

data = hf.phys_signals[useful_channels,:]
# MNE wants EEG and ECG in Volts
for jj,ii in enumerate(useful_channels):
    unit = hf.physical_dimensions[ii]
    if unit == 'uV':
        data[jj,:] = data[jj,:]/1000000
    if unit == 'mV':
        data[jj,:] = data[jj,:]/1000
        
print(data.shape)
customraw = mne.io.RawArray(data, info)
customraw.save('test.fif', overwrite=True)


customraw.plot()  # browse data
