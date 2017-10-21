import pyaudio
import numpy as np
# from matplotlib import pyplot as plt
# import librosa
import threading
import pickle
import librosa
import sklearn
from sklearn.model_selection import GridSearchCV

SR = 22050
CHUNKSIZE = SR/2

valence_model = pickle.load( open( "models_1/valence_test_1.sav", "rb" ) )
arousal_model = pickle.load( open( "models_1/arousal_test_1.sav", "rb" ) )

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SR, input=True, frames_per_buffer=CHUNKSIZE)

# do this as long as you want fresh samples
data = stream.read(CHUNKSIZE)
numpydata = np.fromstring(data, dtype=np.float32)

while data != '':
    mfcc = librosa.feature.mfcc(numpydata, SR, n_mfcc=20)
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    mfcc_feature = np.concatenate(
        (np.mean(mfcc, axis=1), np.std(mfcc, axis=1),  # mfcc feature taken from transfer learning paper
         np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
         np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), axis=0)
    print(valence_model.predict(mfcc_feature))
    print(arousal_model.predict(mfcc_feature))


# close stream
stream.stop_stream()
stream.close()
p.terminate()