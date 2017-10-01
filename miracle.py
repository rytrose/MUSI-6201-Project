import pyaudio
import numpy as np
# from matplotlib import pyplot as plt
from python_speech_features import mfcc
# import librosa
import threading

CHUNKSIZE = 1024

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=CHUNKSIZE)

# do this as long as you want fresh samples
data = stream.read(CHUNKSIZE)
numpydata = np.fromstring(data, dtype=np.int16)

while data != '':
    mfcc_feat = mfcc(numpydata, samplerate=44100, nfft=1024)
    print(mfcc_feat[0])
    data = stream.read(CHUNKSIZE)
    numpydata = np.fromstring(data, dtype=np.int16)

# close stream
stream.stop_stream()
stream.close()
p.terminate()






