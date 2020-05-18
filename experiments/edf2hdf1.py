# this version depends on my own python-edf
"""
basic script to convert an EDF or EDF+ file to our custom 
eeg hdf5 based format "eeghdf" 

edf2hdf1.py <edffilename.edf> [-o output_file_name]
"""
# installation works with python 3.5, 3.6, 3.7
# baseline use was with anaconda 5.2, 5.3 so numpy and h5py already installed
# pip install edflib arrow
# pip install -e git+https://github.com/eegml/eeghdf#egg=eeghdf

# TODO: FIXME!!! remove the EDF annotations channel, don't need to save that
#                probably

import os.path
import datetime
import pprint

import arrow
import numpy as np

import eeghdf
import edflib


### debugging settings ###
# debug = print if set verbose=True

###

def edf_block_iter_generator(
        edf_file, nsamples, samples_per_chunk, dtype='int32'):
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

    nchan = edf_file.signals_in_file

    # 'int32' will work for int16 as well
    buf = np.zeros((nchan, samples_per_chunk), dtype=dtype)

    nchunks = nsamples // samples_per_chunk
    left_over_samples = nsamples - nchunks * samples_per_chunk

    mark = 0
    for ii in range(nchunks):
        for cc in range(nchan):
            edf_file.read_digital_signal(cc, mark, samples_per_chunk, buf[cc])

        yield (buf, mark, samples_per_chunk)
        mark += samples_per_chunk
        # print('mark:', mark)
    # left overs
    if left_over_samples > 0:
        for cc in range(nchan):
            edf_file.read_digital_signal(cc, mark, left_over_samples, buf[cc])

        yield (buf[:, 0:left_over_samples], mark, left_over_samples)



# based upon edf2hdf2 in nk_database_proj/scripts 
def edf2hdf(fn, outfn='', anonymize=False, verbose=False):
    """
    convert an edf file @fn to hdf5 using fairly straightforward mapping
    if no @outfn is specified, then use the same name as the @fn but change extention to "eeg.h5"

    return True if successful
    
    """

    if not outfn:
        base = os.path.basename(fn)
        base, ext = os.path.splitext(base)

        # outfn = os.path.join(hdf_dir, base)
        outfn = base + '.eeg.h5'
        
        # all the data point related stuff

    with edflib.EdfReader(fn) as ef:

        # read all EDF+ header information in just the way I want it
        header = {
            'file_name': os.path.basename(fn),
            'filetype': ef.filetype,
            'patient_name': ef.patient_name,
            'patientcode': ef.patientcode,
            'gender': ef.gender,

            'signals_in_file': ef.signals_in_file,
            'datarecords_in_file': ef.datarecords_in_file,
            'file_duration_100ns': ef.file_duration_100ns,
            'file_duration_seconds': ef.file_duration_seconds,
            'startdate_date': datetime.date(ef.startdate_year, ef.startdate_month, ef.startdate_day),
            'start_datetime': datetime.datetime(ef.startdate_year, ef.startdate_month, ef.startdate_day,
                                                ef.starttime_hour, ef.starttime_minute, ef.starttime_second),
            'starttime_subsecond_offset': ef.starttime_subsecond,

            'birthdate_date': ef.birthdate_date,
            'patient_additional': ef.patient_additional,
            'admincode': ef.admincode,  # usually the study eg. C13-100
            'technician': ef.technician,
            'equipment': ef.equipment,
            'recording_additional': ef.recording_additional,
            'datarecord_duration_100ns': ef.datarecord_duration_100ns,
        }
        if verbose:
            pprint.pprint(header)
            debug = print
        else:
            def nulfunction(*args,**kwargs):
                return None
            debug = nulfunction

        #  use arrow
        start_datetime = header['start_datetime']

        # end_date_time = datetime.datetime(ef.enddate_year, ef.enddate_month, ef.enddate_day, ef.endtime_hour,
        # ef.endtime_minute, ef.endtime_second) # tz naive
        # end_date_time - start_date_time
        duration = datetime.timedelta(seconds=header['file_duration_seconds'])
        

        # derived information
        birthdate = header['birthdate_date']
        if birthdate:
            age = arrow.get(start_datetime) - arrow.get(header['birthdate_date'])

            debug('predicted age: %s' % age)
            # total_seconds() returns a float
            debug('predicted age (seconds): %s' % age.total_seconds())
        else:
            age = datetime.timedelta(seconds=0)
            birthdate = ''

        if anonymize:
            raise Exception('not implemented')


        # anonymized version if necessary
        header['end_datetime'] = header['start_datetime'] + duration

        ############# signal array information ##################

        # signal block related stuff
        nsigs = ef.signals_in_file

        # again know/assume that this is uniform sampling across signals
        # for each record block 
        fs0 = ef.samplefrequency(0)
        signal_frequency_array = ef.get_signal_freqs()
        # print("signal_frequency_array::\n", repr(signal_frequency_array))
        assert all(signal_frequency_array == fs0)

        nsamples0 = ef.samples_in_file(0)  # samples per channel
        debug('nsigs=%s, fs0=%s, nsamples0=%s\n' % (nsigs, fs0, nsamples0))

        num_samples_per_signal = ef.get_samples_per_signal()  # np array
        # print("num_samples_per_signal::\n", repr(num_samples_per_signal), '\n')
        assert all(num_samples_per_signal == nsamples0)

        file_duration_sec = ef.file_duration_seconds
        #print("file_duration_sec", repr(file_duration_sec))

        # Note that all annotations except the top row must also specify a duration.

        # long long onset; /* onset time of the event, expressed in units of 100
        #                     nanoSeconds and relative to the starttime in the header */

        # char duration[16]; /* duration time, this is a null-terminated ASCII text-string */

        # char annotation[EDFLIB_MAX_ANNOTATION_LEN + 1]; /* description of the
        #                             event in UTF-8, this is a null term string of max length 512*/

        # start("x.y"), end, char[20]
        # annotations = ef.read_annotations_as_array() # get numpy array of
        # annotations
        annotations_b = ef.read_annotations_b_100ns_units()
        # print("annotations_b::\n")
        # pprint.pprint(annotations_b)  # get list of annotations

        signal_text_labels = ef.get_signal_text_labels()
        debug("signal_text_labels::\n")
        if verbose:
            pprint.pprint(signal_text_labels)
        
        # ef.recording_additional

        # print()
        signal_digital_mins = np.array(
            [ef.digital_min(ch) for ch in range(nsigs)])
        signal_digital_total_min = min(signal_digital_mins)

        #print("digital mins:", repr(signal_digital_mins))
        #print("digital total min:", repr(signal_digital_total_min))

        signal_digital_maxs = np.array(
            [ef.digital_max(ch) for ch in range(nsigs)])
        signal_digital_total_max = max(signal_digital_maxs)
      
        #print("digital maxs:", repr(signal_digital_maxs))
        #print("digital total max:", repr(signal_digital_total_max))

        signal_physical_dims = [
            ef.physical_dimension(ch) for ch in range(nsigs)]
        # print('signal_physical_dims::\n')
        # pprint.pprint(signal_physical_dims)
        #print()

        signal_physical_maxs = np.array(
            [ef.physical_max(ch) for ch in range(nsigs)])

        #print('signal_physical_maxs::\n', repr(signal_physical_maxs))

        signal_physical_mins = np.array(
            [ef.physical_min(ch) for ch in range(nsigs)])

        #print('signal_physical_mins::\n', repr(signal_physical_mins))

        # this don't seem to be used much so I will put at end
        signal_prefilters = [ef.prefilter(ch).strip() for ch in range(nsigs)]
        #print('signal_prefilters::\n')
        # pprint.pprint(signal_prefilters)
        #print()
        signal_transducers = [ef.transducer(ch).strip() for ch in range(nsigs)]
        #print('signal_transducers::\n')
        #pprint.pprint(signal_transducers)

        with eeghdf.EEGHDFWriter(outfn, 'w') as eegf:
            if header['birthdate_date']:
                birthdate_isostring = header['birthdate_date'].strftime('%Y-%m-%d')
            else:
                birthdate_isostring = ''
                
            eegf.write_patient_info(patient_name=header['patient_name'],
                                    patientcode=header['patientcode'],
                                    gender=header['gender'],
                                    birthdate_isostring=birthdate_isostring,
                                    # gestational_age_at_birth_days
                                    # born_premature
                                    patient_additional=header['patient_additional'])


            rec = eegf.create_record_block(record_duration_seconds=header['file_duration_seconds'],
                                           start_isodatetime=str(header['start_datetime']),
                                           end_isodatetime=str(header['end_datetime']),
                                           number_channels=header['signals_in_file'],
                                           num_samples_per_channel=nsamples0,
                                           sample_frequency=fs0,
                                           signal_labels=signal_text_labels,
                                           signal_physical_mins=signal_physical_mins,
                                           signal_physical_maxs=signal_physical_maxs,
                                           signal_digital_mins=signal_digital_mins,
                                           signal_digital_maxs=signal_digital_maxs,
                                           physical_dimensions=signal_physical_dims,
                                           patient_age_days=age.total_seconds() / 86400.0,
                                           signal_prefilters=signal_prefilters,
                                           signal_transducers=signal_transducers,
                                           technician=header['technician'])

            eegf.write_annotations_b(annotations_b)  # may be should be called record annotations

            edfblock_itr = edf_block_iter_generator(
                ef,
                nsamples0,
                100 * ef.samples_in_datarecord(0)*header['signals_in_file'], # samples_per_chunk roughly 100 datarecords at a time
                dtype='int32')

            signals = eegf.stream_dig_signal_to_record_block(rec, edfblock_itr)
    
        return True # we succeeded



if __name__ == '__main__':
    import sys
    import argparse


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file_name',  help='an edf file name')
    parser.add_argument('--output', '-o', action='store', dest='output_file_name', help='output file name')
    parser.add_argument('-v','--verbosity', action="count", help='increase output verbosity for debug')
    
    args = parser.parse_args()
    # print(args)
        
    edf2hdf(args.input_file_name, args.output_file_name, verbose=args.verbosity)
    
    

                         
