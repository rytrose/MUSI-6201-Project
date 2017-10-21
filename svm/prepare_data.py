import os
import scipy
from scipy.io import wavfile
import numpy as np
import librosa
import csv
import h5py
import cPickle as pickle

LENGTH_OF_SLICES = .5  # length of audio slices in timw
PATH_TO_AUDIO = os.path.abspath(os.path.join('.', os.pardir)) + '/DEAM_audio'
PATH_TO_AROUSAL_LABELS = os.path.abspath(
    os.path.join('.', os.pardir)) + '/DEAM_annotations/annotations/dynamic/arousal.csv'
PATH_TO_VALENCE_LABELS = os.path.abspath(
    os.path.join('.', os.pardir)) + '/DEAM_annotations/annotations/dynamic/valence.csv'

# load audio and chunk into secon-long sections -> calculates mfcc for each chunk -> saves into h5py along with
# corresponding valence arousal labels
with open(PATH_TO_AROUSAL_LABELS, 'rb') as arousal_csv:
    with open(PATH_TO_VALENCE_LABELS, 'rb') as valence_csv:
        arousal_reader = csv.reader(arousal_csv, delimiter=',')
        valence_reader = csv.reader(valence_csv, delimiter=',')
        arousal_reader.next()  # get rid of first row (headers)
        valence_reader.next()  # get rid of first row (headers)
        '''for i in range(1750): # comment out block to test end of file list
            arousal_reader.next()
            valence_reader.next()'''
        valence_counter = sum(1 for line in open(PATH_TO_VALENCE_LABELS))
        final_mfcc = []
        final_arousal = []
        final_valence = []
        final_filenames = []

        progress_counter = 0
        for arousal, valence in zip(arousal_reader, valence_reader):
            print('splitting song ' + str(progress_counter) + ' of ' + str(valence_counter))
            progress_counter = progress_counter + 1
            filename = str(arousal[0]) + '.mp3'  # load examples from DEAM dataset as mp3, use arousal[0] because first
            # entry will be file number
            y, sr = librosa.load(PATH_TO_AUDIO + '/' + filename)
            song_array = np.array(y)
            num_samps = len(song_array)
            slice_length_in_samps = int(round(LENGTH_OF_SLICES * sr))
            extra_samps = num_samps % slice_length_in_samps
            samps_to_add = slice_length_in_samps - extra_samps
            song_array = np.concatenate([song_array, np.zeros(samps_to_add)])
            split = []
            num_slices = len(song_array) / slice_length_in_samps
            for i in range(num_slices):
                curr_split = []
                for samp_offset in range(slice_length_in_samps):
                    curr_index = (i * slice_length_in_samps) + samp_offset
                    curr_split.append(song_array[curr_index])
                split.append(curr_split)
            mfccs = []
            valence_out = []
            arousal_out = []
            try:
                for split_i in range(len(valence) - 1):  # subtract 1 because first entry in file will be file number
                    valence_out.append(float(valence[split_i + 1])) # add back in 1 here since we want to start
                    # indexing at 1
                    arousal_out.append(float(arousal[split_i + 1]))
                    section = np.array(split[split_i])
                    mfcc = librosa.feature.mfcc(section, sr, n_mfcc=20)
                    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
                    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
                    mfcc_feature = np.concatenate(
                        (np.mean(mfcc, axis=1), np.std(mfcc, axis=1),  # mfcc feature taken from transfer learning paper
                         np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                         np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), axis=0)
                    mfccs.append(mfcc_feature)
                final_mfcc.append(mfccs)
                final_valence.append(valence_out)
                final_arousal.append(arousal_out)
                print(filename)
                final_filenames.append(filename)
            except:
                pass
        pickle.dump(final_mfcc, open( "mfcc.p", "wb" ))
        pickle.dump(final_valence, open("valence.p", "wb"))
        pickle.dump(final_arousal, open("arousal.p", "wb"))
        pickle.dump(final_filenames, open("filename.p", "wb"))
