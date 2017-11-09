import librosa
import numpy as np
import os
import cPickle as pickle
import matplotlib.pyplot as plt

PATH_TO_AUDIO = os.path.abspath(os.path.join('.', os.pardir)) + '/DEAM_audio'
filenames = pickle.load( open( "full_one_sec/filename.p", "rb" ) )  # shape (song no)
arousal = pickle.load( open( "full_one_sec/arousal.p", "rb" ) ) #shape (song no, second index)
valence = pickle.load( open( "full_one_sec/valence.p", "rb" ) ) #shape (song no, second index)
arousal_mean = pickle.load( open( "arousal_mean.p", "rb" ) ) #shape (song no, second index)
valence_mean = pickle.load( open( "valence_mean.p", "rb" ) ) #shape (song no, second index)

def prepare_audio(names, path="full_one_sec/audio.p"):
    songs = {}

    for i, filename in enumerate(names):
        y, sr = librosa.load(PATH_TO_AUDIO + '/' + filename)
        song_array = np.array(y)
        songs[filename] = song_array

        print 'Loaded song', str(i), 'out of', str(len(names))
        if sr != 22050:
            print '!!!!!!!!WARNING: SR is', str(sr), 'NOT 22050:', filename

    pickle.dump(songs, open(path, "wb"))



# Takes in a filename of a song in DEAM dataset (all filenames are in filename.p)
# Takes in a split length (in seconds) to divide the song into chunks of that size
# Returns the audio chunks that have associated valence and arousal values from the DEAM data set
def split_audio(filename, split_length_in_seconds, audio=None):
    if(not audio is None):
        song_array = audio
        sr = 22050
    else:
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

# split each file fn in input_filenames with split_audio(fn, split_length_in_seconds)
# and returns splits, arousal, valence
def split_set(input_filenames, split_length_in_seconds, audio):
    split = []
    arousal = []
    valence = []
    for filename in input_filenames:
        this_split, this_arousal, this_valence = split_audio(filename, split_length_in_seconds, audio[filename])
        split.extend(this_split)
        arousal.extend(this_arousal)
        valence.extend(this_valence)
    return split, arousal, valence


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

def plotValenceArousal():

    for i in range(len(filenames)):
        aro = np.array(arousal[i])
        val = np.array(valence[i])

        plt.xlabel('Valence')
        plt.ylabel('Arousal')
        plt.title('Valence/Arousal of ' + str(filenames[i]))
        plt.plot(val, aro, 'ro')
        plt.axis([-1, 1, -1, 1])
        plt.savefig('full_one_sec/val_aro_plots/' + filenames[i][:-4] + '_val_aro_plot.png')
        plt.gcf().clear()
        print 'Finished ' + str(i) + ' out of ' + str(len(filenames))


# plotValenceArousal()

def aggregateValues(inputValues, outputFilename):
    aggregated = np.zeros(len(inputValues))

    for i in range(len(inputValues)):
        values = np.array(inputValues[i])
        mean = np.mean(values)
        aggregated[i] = mean

    print len(inputValues)
    print aggregated.shape
    pickle.dump(aggregated, open(outputFilename + ".p", "wb"))

# aggregateValues(arousal, "arousal_mean")
# aggregateValues(valence, "valence_mean")

def getBalancedFiles(numberFilesPerDimPerBin, exclude=[]):
    exclude_files = -1
    while exclude_files == -1:
        exclude = [item for items in exclude for item in items]
        try:
            exclude_files = set(exclude)
        except:
            print "flattened"


    arousal_n, arousal_bins, arousal_patches = plt.hist(np.array(arousal_mean))
    valence_n, valence_bins, valence_patches = plt.hist(np.array(valence_mean))

    indices = np.arange(len(filenames))
    np.random.shuffle(indices)

    numBins = len(arousal_bins - 1)
    arousal_files = [[] for i in range(numBins - 1)]
    valence_files = [[] for i in range(numBins - 1)]

    i = 0
    while i < len(indices):
        aro_bin_i = np.digitize(arousal_mean[indices[i]], arousal_bins[:-1]) - 1
        val_bin_i = np.digitize(valence_mean[indices[i]], valence_bins[:-1]) - 1

        if len(arousal_files[aro_bin_i]) < numberFilesPerDimPerBin and (not filenames[indices[i]] in exclude_files):
            arousal_files[aro_bin_i].append(filenames[indices[i]])

        if len(valence_files[val_bin_i]) < numberFilesPerDimPerBin and (not filenames[indices[i]] in exclude_files):
            valence_files[val_bin_i].append(filenames[indices[i]])
        i += 1

    return arousal_files, valence_files

# abstract the level of list one level turns [[[1],[2]],[[3],[4]]] -> [[1],[2],[3],[4]]
def abstract_list(the_list):
    flat_list = [item for sublist in the_list for item in sublist]
    return flat_list

# creates train_set using getBalancedFiles(num_train) and test set using getBalancedFiles(num_test, train_set)
# outputs audio.p using
def make_train_and_test_sets(num_train, num_test, path="train_test_sets/"):
    train_arousal_filenames, train_valence_filenames = getBalancedFiles(num_train)
    test_arousal_filenames, test_valence_filenames = getBalancedFiles(num_test, [train_arousal_filenames, train_valence_filenames])

    train_arousal_filenames = abstract_list(train_arousal_filenames)
    train_valence_filenames = abstract_list(train_valence_filenames)
    test_arousal_filenames = abstract_list(test_arousal_filenames)
    test_valence_filenames = abstract_list(test_valence_filenames)

    files = set(train_arousal_filenames + train_valence_filenames + test_arousal_filenames + test_valence_filenames)
    audio = {}
    for i, file in enumerate(files):
        y, sr = librosa.load("../DEAM_audio/" + file)
        audio[file] = np.array(y)
        print "loaded " + str(i) + " of " + str(len(files))

    dir_name = "train-" + str(num_train) + "_test-" + str(num_test) + "/"
    os.mkdir(path + dir_name)

    pickle.dump(audio, open(path + dir_name + "audio.p", "wb"))
    pickle.dump(train_arousal_filenames, open(path + dir_name + "train_arousal_filenames.p", "wb"))
    pickle.dump(train_valence_filenames, open(path + dir_name + "train_valence_filenames.p", "wb"))
    pickle.dump(test_arousal_filenames, open(path + dir_name + "test_arousal_filenames.p", "wb"))
    pickle.dump(test_valence_filenames, open(path + dir_name + "test_valence_filenames.p", "wb"))

# prepare_audio(filenames)

# l1, l2 = getBalancedFiles(2)
# print l1, l2
# l3, l4 = getBalancedFiles(2, [l1, l2])
# print l3, l4
# print getBalancedFiles(2, [l1, l2, l3, l4])

def calcMFCCs(audio):
    final_feature = []
    for section in audio:
        mfcc = librosa.feature.mfcc(np.array(section), 22050, n_mfcc=20)
        dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
        ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
        mfcc_feature = np.concatenate(
            (np.mean(mfcc, axis=1), np.std(mfcc, axis=1),  # mfcc feature taken from transfer learning paper
             np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
             np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), axis=0)
        final_feature.append(mfcc_feature)

    return final_feature