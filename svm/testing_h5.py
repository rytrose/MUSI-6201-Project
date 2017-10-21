import cPickle as pickle

mfcc = pickle.load( open( "subset/mfcc.p", "rb" ) ) # shape (song no, second index, feature len)
arousal = pickle.load( open( "subset/arousal.p", "rb" ) ) #shape (song no, second index)
valence = pickle.load( open( "subset/valence.p", "rb" ) ) #shape (song no, second index)
filename = pickle.load( open( "subset/filename.p", "rb" ) )  # shape (song no)