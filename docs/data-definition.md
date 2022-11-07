# eeghdf protocol

Patient
-------
patient_name : str : convention is "last_name first_name second_name..."
patientcode: str : can be used for manythings
gender: str : typically biological sex at birth suggestions F-female M-male U-unknown I-indeterminate 
birthdate_isostring = str : isodate format YYYY-MM-DD
gestational_age_at_birth_days :  float=-1.0  : GA is number of days since last menstrual period -1.0 is sentinal value for unknown
born_premature : str="unknown" : enum  {"true" "false" "unknown"}
patient_addition : str="" : adopted from EDF for anything you want to add



Record
see ```eeghdf.writer.EEGHDFWriter.create_record_block``


stored in hf['record-0'].attrs
-------------------------------
[('bits_per_sample', 16),
 ('end_isodatetime', '2011-11-11 12:17:06'),
 ('number_channels', 21),
 ('number_samples_per_channel', 1012480),
 ('patient_age_days', 0.0),
 ('sample_frequency', 256.0),
 ('start_isodatetime', '2011-11-11 11:11:11'),
 ('studyadmincode', "75"),
 ('technician', '')]


bits_per_sample : int; 
end_isodatetime : str; iso-datetime  string
number_channels : int;  
num_samples_per_channel : int; look up this one
sample_frequency : float; Hz
patient_age_days : float; auto calculate float form patienet info?
start_isodatetime : str;  isodate time yyyy-MM-DD'T'HH:mm:ss ex: 2000-10-31T01:30:00  may hadd decimal 01:30:00.000
studyadmincode : str=""; arbitrary identifying strings
technician : str="" ; often information identifying the tech setting up study 


record_duration_seconds: float
signal_labels

signal_physical_mins
signal_physical_maxs
signal_digital_mins
signal_digital_maxs
physical_dimensions
signal_prefilters=None
signal_transducers=None
edf_annotations

[('dense_seizure_annotations',
  <HDF5 group "/record-0/dense_seizure_annotations" (1 members)>),
 ('edf_annotations', <HDF5 group "/record-0/edf_annotations" (3 members)>),
 ('physical_dimensions',
  <HDF5 dataset "physical_dimensions": shape (21,), type "|O">),
 ('prefilters', <HDF5 dataset "prefilters": shape (21,), type "|O">),
 ('signal_digital_maxs',
  <HDF5 dataset "signal_digital_maxs": shape (21,), type "<i8">),
 ('signal_digital_mins',
  <HDF5 dataset "signal_digital_mins": shape (21,), type "<i8">),
 ('signal_labels', <HDF5 dataset "signal_labels": shape (21,), type "|O">),
 ('signal_physical_maxs',
  <HDF5 dataset "signal_physical_maxs": shape (21,), type "<f8">),
 ('signal_physical_mins',
  <HDF5 dataset "signal_physical_mins": shape (21,), type "<f8">),
 ('signals', <HDF5 dataset "signals": shape (21, 1012480), type "<i2">),
 ('transducers', <HDF5 dataset "transducers": shape (21,), type "|O">)]



 Extensions:
 ===========
 The current defintion for EEGs handles well edf-like data as long as it has uniform sampling
 but of course extensions to the base set of data will be needed.

 The idea is to try out extensions to the protocol by adding an "extensions" group.
 A "description" attribute should be added to each subgroup to explain what the extension is. 

 This is best explained by a specific example. Here we extend the eeghdf ver 3 protocol to 
 include the data for the Stevenson et al. (2019?) neonatal dataset
```
$ h5ls eeg75.annot.eeg.h5 
patient                  Group
record-0                 Group

$ h5ls eeg75.annot.eeg.h5/record-0
edf_annotations          Group
extensions               Group   <- this "extensions" group was added
physical_dimensions      Dataset {21}
prefilters               Dataset {21}
signal_digital_maxs      Dataset {21}
signal_digital_mins      Dataset {21}
signal_labels            Dataset {21}
signal_physical_maxs     Dataset {21}
signal_physical_mins     Dataset {21}
signals                  Dataset {21, 1012480}
transducers              Dataset {21}

$ h5ls eeg75.annot.eeg.h5/record-0/extensions
dense_seizure_annotations Group

$ h5ls eeg75.annot.eeg.h5/record-0/extensions/dense_seizure_annotations
annotarr                 Dataset {3, 3955}
consensus_labels         Dataset {3955}
consensus_sz_labels      Dataset {3955}


$ h5ls  -r eeg75.annot.eeg.h5
/                        Group
/patient                 Group
/patient/extensions      Group
/record-0                Group
/record-0/edf_annotations Group
/record-0/edf_annotations/durations_char16 Dataset {0}
/record-0/edf_annotations/starts_100ns Dataset {0}
/record-0/edf_annotations/texts Dataset {0}
/record-0/extensions     Group
/record-0/extensions/dense_seizure_annotations Group
/record-0/extensions/dense_seizure_annotations/annotarr Dataset {3, 3955}
/record-0/extensions/dense_seizure_annotations/consensus_labels Dataset {3955}
/record-0/extensions/dense_seizure_annotations/consensus_sz_labels Dataset {3955}
/record-0/physical_dimensions Dataset {21}
/record-0/prefilters     Dataset {21}
/record-0/signal_digital_maxs Dataset {21}
/record-0/signal_digital_mins Dataset {21}
/record-0/signal_labels  Dataset {21}
/record-0/signal_physical_maxs Dataset {21}
/record-0/signal_physical_mins Dataset {21}
/record-0/signals        Dataset {21, 1012480}
/record-0/transducers    Dataset {21}
```