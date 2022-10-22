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
 ('studyadmincode', 75),
 ('technician', '')]


start_isodatetime : str : isodate time yyyy-MM-DD'T'HH:mm:ss ex: 2000-10-31T01:30:00  may hadd decimal 01:30:00.000
end_isodatetime : str : iso-datetime  string
number_channels : int 
num_samples_per_channel
sample_frequency
patient_age_days  # auto calculate float form patienet info?

bits_per_sample : int
technician : str="" 
studyadmincode : str=""

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