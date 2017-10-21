import os
import sys
import numpy as np
import librosa
import time
import sklearn
import pdb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import logging
import cPickle as pickle
import multiprocessing

class OptionalStandardScaler(StandardScaler): # class taken from transfer learning paper
    def __init__(self, on=False):
        super(OptionalStandardScaler, self).__init__(with_mean=True, with_std=True)

# abstract the level of list one level turns [[[1],[2]],[[3],[4]]] -> [[1],[2],[3],[4]]
def abstract_list(the_list):
    flat_list = [item for sublist in the_list for item in sublist]
    return flat_list

#load the data
print 'loading mfcc'
mfcc = pickle.load( open( "subset/mfcc.p", "rb" ) ) # shape (song no, second index, feature len)
print 'loading arousal'
arousal = pickle.load( open( "subset/arousal.p", "rb" ) ) #shape (song no, second index)
print 'loading valence'
valence = pickle.load( open( "subset/valence.p", "rb" ) ) #shape (song no, second index)
print 'loading filename'
filename = pickle.load( open( "subset/filename.p", "rb" ) )  # shape (song no)

print 'flattening mfcc'
flat_mfcc = abstract_list(mfcc)
print 'flattening arousal'
flat_arousal = abstract_list(arousal)
print 'flattening valence'
flat_valence = abstract_list(valence)


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
num_examples = len(flat_mfcc)
subset_id = round(num_examples/10)# index for subset of data used for testing

clf = GridSearchCV(pipe, params, cv=None, n_jobs=n_jobs, pre_dispatch='8*n_jobs', verbose=10).fit(flat_mfcc, flat_arousal)
filename = 'arousal_test_2.sav'
pickle.dump(clf, open(filename, 'wb'))