import librosa
import numpy as np
import os
import cPickle as pickle

PATH_TO_AUDIO = os.path.abspath(os.path.join('.', os.pardir)) + '/DEAM_audio'

def split_audio(filename, split_length_in_seconds):
    y, sr = librosa.load(PATH_TO_AUDIO + '/' + filename)
    song_array = np.array(y)
    num_samps = len(song_array)
    slice_length_in_samps = int(round(split_length_in_seconds * sr))
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
    return split, sr

#TEST AUDIO SPLITTING
#filename = pickle.load( open( "full_one_sec/filename.p", "rb" ) )
#split, sr = split_audio(filename[0], .5)
#librosa.output.write_wav( os.path.abspath('.') + '/fuck_MIR.wav', np.array(split[5]), sr)