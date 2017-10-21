import sys, os.path
import librosa
import cPickle as pickle
import numpy as np
import csv
transfer_learning_music_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/transfer_learning_music/')
sys.path.append(transfer_learning_music_dir)
import easy_feature_extraction

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
        for i in range(1800): # comment out block to test end of file list
            arousal_reader.next()
            valence_reader.next()
        valence_counter = sum(1 for line in open(PATH_TO_VALENCE_LABELS))
        final_mfcc = []
        final_arousal = []
        final_valence = []
        final_filenames = []
        final_convnet_15 = []
        final_convnet_235 = []

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
            convnet_15 = []
            convnet_235 = []
            valence_out = []
            arousal_out = []
            #try:
            for split_i in range(len(valence) - 1):  # subtract 1 because first entry in file will be file number
                valence_out.append(float(valence[split_i + 1])) # add back in 1 here since we want to start
                # indexing at 1
                arousal_out.append(float(arousal[split_i + 1]))
                section = np.array(split[split_i])
                mfcc = librosa.feature.mfcc(section, sr, n_mfcc=20)
                dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
                ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
                convnet_feature_15, convnet_feature_235 = easy_feature_extraction.calc(section, LENGTH_OF_SLICES)
                convnet_feature_15 = np.concatenate(convnet_feature_15, axis=0)
                convnet_feature_235 = np.concatenate(convnet_feature_235, axis=0)
                print convnet_feature_15
                convnet_15.append(convnet_feature_15)
                convnet_235.append(convnet_feature_235)
                #mfcc_feature = np.concatenate(
                #    (np.mean(mfcc, axis=1), np.std(mfcc, axis=1),  # mfcc feature taken from transfer learning paper
                #     np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                #     np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), axis=0)
                #mfccs.append(mfcc_feature)

            final_mfcc.append(mfccs)
            final_valence.append(valence_out)
            final_arousal.append(arousal_out)
            final_convnet_15.append(convnet_15)
            final_convnet_235.append(convnet_235)
            print(filename)
            final_filenames.append(filename)
            '''except:
                print "Unexpected error:", sys.exc_info()[0]
                pass'''
        print final_convnet_15
        print len(final_convnet_15)
        print len(len(final_convnet_15))
        pickle.dump(final_convnet_15, open( "convnet15_arousal.p", "wb" ))
        pickle.dump(final_convnet_235, open("convnet235_valence.p", "wb"))
