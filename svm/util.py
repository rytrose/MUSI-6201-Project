import librosa
import numpy as np
import os
import cPickle as pickle
import matplotlib.pyplot as plt
import sys
sys.path.append("./../transfer_learning_music")
import easy_feature_extraction
import multiprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

PATH_TO_AUDIO = os.path.abspath(os.path.join('.', os.pardir)) + '/DEAM_audio'
filenames = pickle.load( open( "full_one_sec/filename.p", "rb" ) )  # shape (song no)
arousal = pickle.load( open( "full_one_sec/arousal.p", "rb" ) ) #shape (song no, second index)
valence = pickle.load( open( "full_one_sec/valence.p", "rb" ) ) #shape (song no, second index)
arousal_mean = pickle.load( open( "arousal_mean.p", "rb" ) ) #shape (song no, second index)
valence_mean = pickle.load( open( "valence_mean.p", "rb" ) ) #shape (song no, second index)

class OptionalStandardScaler(StandardScaler): # class taken from transfer learning paper
    def __init__(self, on=False):
        super(OptionalStandardScaler, self).__init__(with_mean=True, with_std=True)

# Buffers and serializes audio for later use
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

# trim timestamps of audio according to split sections
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

# associates the latest AV label to a given timestamp
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

# computes and stores the mean AV values
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

# Splits the dataset into ten histogram bins of average arousal/valence value
# Returns a subset of files distributed across the histogram
# Use the optional exclude parameter to get unique sets on iterative calls
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

# Creates train_set using getBalancedFiles(num_train) and test set using getBalancedFiles(num_test, train_set)
def make_train_and_test_sets(num_train, num_test, path="train_test_sets/", exclude=[]):
    train_arousal_filenames, train_valence_filenames = getBalancedFiles(num_train, exclude)
    test_arousal_filenames, test_valence_filenames = getBalancedFiles(num_test, [train_arousal_filenames, train_valence_filenames, exclude])

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
    os.mkdir(os.getcwd() + '/' + path + dir_name[:-1])

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

# Calculates the MFCC feature for a list of chunks of audio
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

# Calculates the convnet feature for a list of chunks of audio
# Since the convnet feature extraction requires 29s of audio, for
#  any set of audio less than 29s this algorithm will repeat the
#  input audio as many times as possible until it fits into 29s
#  (based on suggestion from transfer learning paper)
def calcConvnetFeatures(audio):
    final_feature = []
    feature_samples = 12000 * 29
    feature_buffer = np.zeros(feature_samples)

    for i in range(len(audio)):
        section = audio[i]
        sample_index = 0
        np_audio = librosa.core.resample(np.array(section), 22050, 12000)

        while sample_index < len(np_audio):
            # Audio left is less than 29s, repeat the chunk we have and predict
            if len(np_audio[sample_index:len(np_audio)]) < len(feature_buffer):
                feature_index = 0
                while feature_index < len(feature_buffer):
                    # Repeat the audio left
                    end_index = min(feature_index + len(np_audio), len(feature_buffer)) - feature_index
                    feature_buffer[feature_index:min(feature_index + len(np_audio), len(feature_buffer))] = np_audio[0:end_index]
                    feature_index += len(np_audio)
                # Predict
                #print "FEAT BUFF", np.shape(feature_buffer)
                feature = easy_feature_extraction.extractFeatures(feature_buffer)
                final_feature.append(feature)

                # Reset feature_buffer
                feature_buffer = np.zeros(feature_samples)

            # Audio left is greater than 29s
            else:
                feature_buffer = np_audio[sample_index:sample_index + len(feature_buffer)]

                # Predict
                feature = easy_feature_extraction.extractFeatures(feature_buffer)
                final_feature.append(feature)

                # Reset feature_buffer
                feature_buffer = np.zeros(feature_samples)

            # Iterate
            sample_index += len(feature_buffer)

    final_feature = [np.concatenate(np.array(l)) for l in final_feature]
    return final_feature


# This will take feature vectors generated by calls to split_audio (with lengths in multiples of two)
#  and filter and concatenate the feature vectors appropriately such that
#  the shortest label vector dictates the length of the final feature and label vectors
# Used for forward selection
def trim(feature_list, label_list):
    feature_vector = []
    label_vector = []

    shortest_length = min([len(l) for l in label_list])
    for l in label_list:
        if len(l) == shortest_length:
            label_vector = l

    # For every feature in the shortest list, concatenate the
    # appropriate feature from each list
    for feature_index in range(len(label_vector)):
        current_feature = np.array([])

        for list_index in range(len(label_list)):
            this_list = feature_list[list_index]
            this_list_length = len(this_list)
            final_index = ((feature_index + 1) * int(this_list_length / shortest_length)) - 1
            current_feature = np.concatenate((current_feature, this_list[final_index]))

        feature_vector.append(current_feature)

    return feature_vector, label_vector

BIG_BOY = 999999999999

# Standard forward selection algorithm
# Uses either "r2" (r-squared) or "mse" (mean squared error)
#  as the evaluation metric
def forward_select(candidate_svms, train_feature_vectors, train_label_vectors, test_feature_vectors, test_label_vectors, metric="mse"):
    selected_feature_indexes = []
    best_selected_feature_indexes = []
    remaining_feature_indexes = [i for i in range(len(train_label_vectors))]
    best_svm = candidate_svms[0]
    best_performance = -1 * BIG_BOY
    while len(remaining_feature_indexes) > 0:
        best_new_feature = -1
        best_new_feature_performance = -1 * BIG_BOY
        best_new_svm = candidate_svms[0]
        for trial_feat_index in remaining_feature_indexes:
            trial_selected_feature_indexes = selected_feature_indexes + [trial_feat_index]
            trial_selected_train_features = []
            trial_selected_train_labels = []
            trial_selected_test_features = []
            trial_selected_test_labels = []
            for trial_selected_feature_index in trial_selected_feature_indexes:
                trial_selected_train_features.append(train_feature_vectors[trial_selected_feature_index])
                trial_selected_train_labels.append(train_label_vectors[trial_selected_feature_index])
                trial_selected_test_features.append(test_feature_vectors[trial_selected_feature_index])
                trial_selected_test_labels.append(test_label_vectors[trial_selected_feature_index])
            trial_selected_train_features, trial_selected_train_labels = trim(trial_selected_train_features,
                                                                              trial_selected_train_labels)
            trial_selected_test_features, trial_selected_test_labels = trim(trial_selected_test_features,
                                                                              trial_selected_test_labels)
            best_trial_svm = candidate_svms[0]
            best_trial_svm_performance = -1 * BIG_BOY
            for trial_svm in candidate_svms:
                trial_svm = trial_svm.best_estimator_.fit(trial_selected_train_features, trial_selected_train_labels)
                predictions = trial_svm.predict(trial_selected_test_features)
                expected = trial_selected_test_labels
                if metric == "r2":
                    trial_svm_performance =  ((np.corrcoef(predictions, expected)[0, 1]) ** 2) # r-squared value
                elif metric == "r":
                    trial_svm_performance = (np.corrcoef(predictions, expected)[0, 1])  # r value
                else:
                    trial_svm_performance = -1 * np.mean((expected - predictions) ** 2) # mean squared error
                if trial_svm_performance > best_trial_svm_performance:
                    best_trial_svm_performance = trial_svm_performance
                    best_trial_svm = trial_svm

            if best_trial_svm_performance > best_new_feature_performance:
                best_new_feature = trial_feat_index
                best_new_feature_performance = best_trial_svm_performance
                best_new_svm = best_trial_svm

        if best_new_feature_performance > best_performance:
            remaining_feature_indexes.remove(best_new_feature)
            selected_feature_indexes.append(best_new_feature)
            best_selected_feature_indexes = [i for i in selected_feature_indexes]
            best_svm = best_new_svm
            best_performance = best_new_feature_performance
        else:
            remaining_feature_indexes.remove(best_new_feature)
            selected_feature_indexes.append(best_new_feature)
        if metric == "mse":
            best_performance = best_performance * -1
    return best_selected_feature_indexes, best_performance, best_svm

# Returns the predictions for a given subset of models
def getPredictionVector(prediction, idxs):
    feat_vec = []
    for i in range(len(prediction[0])):
        current_vec = []
        for idx in idxs:
            current_vec.append(prediction[idx][i])
        feat_vec.append(current_vec)
    return feat_vec

# Standard backward selection algorithm
# Uses either "r2" (r-squared) or "mse" (mean squared error)
#  as the evaluation metric
def backwardSelect(model_names, train_predictions, train_expected, test_predictions, test_expected, metric="mse"):
    remaining_model_indexes = [i for i in range(len(model_names))]
    train_vec = getPredictionVector(train_predictions, remaining_model_indexes)
    test_vec = getPredictionVector(test_predictions, remaining_model_indexes)
    best_hyper_model = train_SVM(train_expected, train_vec)
    current_test_predictions = best_hyper_model.predict(test_vec)
    if metric == "r2":
        best_performance = (np.corrcoef(current_test_predictions, test_expected)[0, 1]) ** 2
    else:
        best_performance = -1 * np.mean((test_expected - current_test_predictions) ** 2)
    while len(remaining_model_indexes) > 0:
        best_trial_performance = -1 * BIG_BOY
        best_trial_index = -1
        best_trial_model = None
        print(remaining_model_indexes)
        for model_index in remaining_model_indexes:
            trial_remaining_indexes = [i for i in remaining_model_indexes]
            trial_remaining_indexes.remove(model_index)
            train_vec = getPredictionVector(train_predictions, trial_remaining_indexes)
            test_vec = getPredictionVector(test_predictions, trial_remaining_indexes)
            trial_model = train_SVM(train_expected, train_vec)
            current_test_predictions = trial_model.predict(test_vec)
            if metric == "r2":
                trial_performance = (np.corrcoef(current_test_predictions, test_expected)[0, 1]) ** 2
            else:
                trial_performance = -1 * np.mean((test_expected - current_test_predictions) ** 2)
            if trial_performance > best_trial_performance:
                best_trial_performance = trial_performance
                best_trial_index = model_index
                best_trial_model = trial_model
        if best_trial_performance > best_performance:
            best_performance = best_trial_performance
            best_hyper_model = best_trial_model
            remaining_model_indexes.remove(best_trial_index)
        else:
            break

    text_file = open("hypermodel/input_model_names.txt", "w")
    input_models = []
    for i in remaining_model_indexes:
        text_file.write(model_names[i] + " ")
        input_models.append(model_names[i])
    text_file.close()
    pickle.dump(best_hyper_model, open("hypermodel/hypermodel.sav", 'wb'))
    return best_hyper_model, input_models

# Given a training set of features and a test set of fetaures
#  normalizes both sets according to the mean and std of the
#  training set
def normalize_sets(train_set, test_set):
    train_mean = np.mean(train_set)
    train_std = np.std(train_set)
    train_set_norm = [(arr - train_mean)/train_std for arr in train_set]
    test_set_norm = [(arr - train_mean) / train_std for arr in test_set]
    return train_set_norm, test_set_norm

# like above func but can take in third set
def norm_with_validate(train_set, test_set, validate_set):
    train_mean = np.mean(train_set)
    train_std = np.std(train_set)
    train_set_norm = [(arr - train_mean)/train_std for arr in train_set]
    test_set_norm = [(arr - train_mean) / train_std for arr in test_set]
    validate_set_norm = [(arr - train_mean) / train_std for arr in validate_set]
    return train_set_norm, test_set_norm, validate_set_norm

# Trains an SVM using grid search given labels and features
def train_SVM(labels, features):
    #train the model
    n_cpu = multiprocessing.cpu_count()
    n_jobs = int(n_cpu * 0.8)

    gp = [{"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['rbf'],
                 "gamma": [0.5 ** i for i in [3, 5, 7, 9, 11, 13]] + ['auto']},
                {"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['linear']}]
    params = []
    for dct in gp:  # should be dict of list for e.g. svm
        sub_params = {'stdd__on': [True, False]}
        sub_params.update({'clf__' + key: value for (key, value) in dct.iteritems()})
        params.append(sub_params)

    print 'training model...'
    estimators = [('stdd', OptionalStandardScaler()), ('clf', SVR())]
    pipe = Pipeline(estimators)
    # cv should equal 10 by default according to transfer learning paper
    num_examples = len(features)
    subset_id = round(num_examples/10)# index for subset of data used for testing

    clf = GridSearchCV(pipe, params, cv=None, n_jobs=n_jobs, pre_dispatch='8*n_jobs', verbose=0)
    clf.fit(features, labels)
    #filename = 'baseline_test_1_arousal.sav'
    #pickle.dump(clf, open(filename, 'wb'))
    return clf

