from util import *
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import cPickle as pickle
import multiprocessing
import threading
import OSC

class OptionalStandardScaler(StandardScaler): # class taken from transfer learning paper
    def __init__(self, on=False):
        super(OptionalStandardScaler, self).__init__(with_mean=True, with_std=True)

class Model:
    def __init__(self):
        self.guiServer = OSC.OSCServer(('127.0.0.1', 7100))
        self.guiServerThread = threading.Thread(target=self.guiServer.serve_forever)
        self.guiServerThread.daemon = False
        self.guiServerThread.start()

        self.guiClient = OSC.OSCClient()
        self.guiClient.connect(('127.0.0.1', 57120))

        self.audio = pickle.load( open( "train_test_sets/train-1_test-1/audio.p", "rb" ) )  # { songfile: [audiodata] }
        print "loaded audio"
        self.train_arousal_filenames = pickle.load( open( "train_test_sets/train-1_test-1/train_arousal_filenames.p", "rb" ) )  # [filenames...]
        self.train_valence_filenames = pickle.load( open( "train_test_sets/train-1_test-1/train_valence_filenames.p", "rb" ) )  # [filenames...]
        self.test_arousal_filenames = pickle.load( open( "train_test_sets/train-1_test-1/test_arousal_filenames.p", "rb" ) )  # [filenames...]
        self.test_valence_filenames = pickle.load( open( "train_test_sets/train-1_test-1/test_valence_filenames.p", "rb" ) )  # [filenames...]

        print "loaded filenames"

    def sendOSCMessage(self, addr, *msgArgs):
        msg = OSC.OSCMessage()
        msg.setAddress(addr)
        msg.append(*msgArgs)
        self.guiClient.send(msg)

    def sendPrediction(self, valence, arousal, confidence):
        self.sendOSCMessage("/setPrediction", [valence, arousal, confidence])

    def baseline(self):
        # split the audio and assign appropriate labels
        self.train_arousal_audio, self.train_arousal_labels, self.dumby_doo_doo = split_set(self.train_arousal_filenames, 0.5, self.audio)
        self.train_valence_audio, self.dumby_doo_doo, self.train_valence_labels = split_set(self.train_valence_filenames, 0.5, self.audio)
        self.test_arousal_audio, self.test_arousal_labels, self.dumby_doo_doo = split_set(self.test_arousal_filenames, 0.5, self.audio)
        self.test_valence_audio, self.dumby_doo_doo, self.test_valence_labels = split_set(self.test_valence_filenames, 0.5, self.audio)

        print('starting mfccs')
        self.train_arousal_mfcc = calcMFCCs(self.train_arousal_audio)
        self.train_valence_mfcc = calcMFCCs(self.train_valence_audio)
        self.test_arousal_mfcc = calcMFCCs(self.test_arousal_audio)
        self.test_valence_mfcc = calcMFCCs(self.test_valence_audio)

        self.clf = pickle.load( open( "baseline_test_1_arousal.sav", "rb" ) )


    def train(self, labels, features, filename):
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
        num_examples = len(self.train_arousal_mfcc)
        subset_id = round(num_examples/10)# index for subset of data used for testing

        clf = GridSearchCV(pipe, params, cv=None, n_jobs=n_jobs, pre_dispatch='8*n_jobs', verbose=10)
        clf.fit(features, labels)
        #filename = 'baseline_test_1_arousal.sav'
        pickle.dump(clf, open(filename, 'wb'))
        return clf