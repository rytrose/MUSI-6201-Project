import librosa
import numpy as np
import os
import cPickle as pickle

PATH_TO_AUDIO = os.path.abspath(os.path.join('.', os.pardir)) + '/DEAM_audio'
filenames = pickle.load( open( "full_one_sec/filename.p", "rb" ) )  # shape (song no)
arousal = pickle.load( open( "full_one_sec/arousal.p", "rb" ) ) #shape (song no, second index)
valence = pickle.load( open( "full_one_sec/valence.p", "rb" ) ) #shape (song no, second index)



def prepare_audio(names):
    songs = {}

    for i, filename in enumerate(names):
        y, sr = librosa.load(PATH_TO_AUDIO + '/' + filename)
        song_array = np.array(y)
        songs[filename] = song_array

        print 'Loaded song', str(i), 'out of', str(len(names))
        if sr != 22050:
            print '!!!!!!!!WARNING: SR is', str(sr), 'NOT 22050:', filename

    pickle.dump(songs, open("full_one_sec/audio.p", "wb"))

prepare_audio(filenames)



# Takes in a filename of a song in DEAM dataset (all filenames are in filename.p)
# Takes in a split length (in seconds) to divide the song into chunks of that size
# Returns the audio chunks that have associated valence and arousal values from the DEAM data set
def split_audio(filename, split_length_in_seconds):
    y, sr = librosa.load(PATH_TO_AUDIO + '/' + filename)
    song_array = np.array(y)
    num_samps = len(song_array)
    slice_length_in_samps = int(round(split_length_in_seconds * sr))
    extra_samps = num_samps % slice_length_in_samps
    samps_to_add = slice_length_in_samps - extra_samps
    song_array = np.concatenate([song_array, np.zeros(samps_to_add)])
    split = []
    timestamps = []
    num_slices = len(song_array) / slice_length_in_samps
    for i in range(num_slices):
        curr_split = []
        for samp_offset in range(slice_length_in_samps):
            curr_index = (i * slice_length_in_samps) + samp_offset
            curr_split.append(song_array[curr_index])
        split.append(curr_split)
        timestamps.append((((1 + i) * slice_length_in_samps) / float(sr)) * 1000);

    song_index = filenames.index(filename)
    song_arousal = arousal[song_index]
    song_valence = valence[song_index]

    label_timestamps = [float((x * 500) + 15000) for x in range(len(song_arousal))]

    trimmed_split, trimmed_timestamps = trim_song_timestamps(split, timestamps, label_timestamps)

    final_arousal, final_valence = associateLabels(trimmed_timestamps, label_timestamps, song_arousal, song_valence)

    return trimmed_split, final_arousal, final_valence

def trim_song_timestamps(split, timestamps, label_timestamps):
    trimmed_split = []
    trimmed_timestamps = []

    min_label_timestamp = label_timestamps[0]
    max_label_timestamp = label_timestamps[-1]

    for i in range(len(timestamps)):
        timestamp = timestamps[i]

        if timestamp >= min_label_timestamp and timestamp <= max_label_timestamp:
            trimmed_split.append(split[i])
            trimmed_timestamps.append(timestamp)

    return trimmed_split, trimmed_timestamps

def associateLabels(trimmed_timestamps, label_timestamps, song_arousal, song_valence):
    final_arousal = []
    final_valence = []

    for i in range(len(trimmed_timestamps)):
        timestamp = trimmed_timestamps[i]
        closest_label_timestamp = min(label_timestamps, key=lambda x:abs(x-timestamp))
        final_arousal.append(song_arousal[label_timestamps.index(closest_label_timestamp)])
        final_valence.append(song_valence[label_timestamps.index(closest_label_timestamp)])

    return final_arousal, final_valence

#TEST AUDIO SPLITTIN
# filename = pickle.load( open( "full_one_sec/filename.p", "rb" ) )
#
# for i in range(5):
#     split, ar, va = split_audio(filename[i + 1780], 0.72)
#     print len(split), len(ar), len(va)
#
# librosa.output.write_wav( os.path.abspath('.') + '/fuck_MIR.wav', np.array(split[5]), sr)