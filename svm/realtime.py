import pyaudio
import OSC
import numpy as np
from util import *
import threading
import os
import pickle

SR = 22050
CHUNKSIZE = SR/2

class Realtime(threading.Thread):
    def __init__(self, final_model_aro, final_model_val):
        threading.Thread.__init__(self)

        # Initialize OSC
        self.guiServer = OSC.OSCServer(('127.0.0.1', 7100))
        self.guiServerThread = threading.Thread(target=self.guiServer.serve_forever)
        self.guiServerThread.daemon = False
        self.guiServerThread.start()

        self.guiClient = OSC.OSCClient()
        self.guiClient.connect(('127.0.0.1', 57120))

        # Initalize an empty 16s buffer for audio
        self.audio_buffer = np.zeros(SR * 16)

        # Will contain predictions from each model
        self.predictions = {}

        # Initialize audio thread
        audio_thread = Audio(self.audio_buffer)
        audio_thread.start()

        # Initialize model threads
        # Initialize convnet models
        for filename in os.listdir(os.getcwd() + '/convnet_models'):
            split = filename.split('_')
            if split[1] == "arousal":
                if not split[3] in ['half', '1']:
                    model_name = "_".join(split[2:5])
                    arousal_model_filename = filename
                    split[1] = "valence"
                    valence_model_filename = "_".join(split)

                    self.predictions[model_name] = [-1, -1]

                    predictor_thread = Predictor(model_name, arousal_model_filename, valence_model_filename,
                                                 self.audio_buffer, self.predictions)
                    predictor_thread.start()

        # Initialize MFCC models
        for filename in os.listdir(os.getcwd() + '/mfcc_models'):
            split = filename.split('_')
            if split[1] == "arousal":
                model_name = "_".join(split[2:5])
                arousal_model_filename = filename
                split[1] = "valence"
                valence_model_filename = "_".join(split)

                self.predictions[model_name] = [-1, -1]

                predictor_thread = Predictor(model_name, arousal_model_filename, valence_model_filename,
                                             self.audio_buffer, self.predictions)
                predictor_thread.start()

        # Initialize final model
        self.final_arousal_model = pickle.load(open("hypermodels/" + final_model_aro, "rb"))
        self.final_valence_model = pickle.load(open("hypermodels/" + final_model_val, "rb"))

    def run(self):
        # Predict and send to the front end
        final_valence_feature = []
        final_arousal_feature = []

        for prediction in self.predictions:
            final_valence_feature += prediction[0]
            final_arousal_feature += prediction[1]

        valence = self.final_valence_model.predict(final_valence_feature)
        arousal = self.final_arousal_model.predict(final_arousal_feature)
        confidence = 0.5 * np.std(np.array(final_arousal_feature)) + 0.5 * np.std(np.array(final_valence_feature))

        print valence, arousal, confidence
        self.sendOSCMessage("/address", [valence, arousal, confidence])

    def sendOSCMessage(self, addr, *msgArgs):
        msg = OSC.OSCMessage()
        msg.setAddress(addr)
        msg.append(*msgArgs)
        self.guiClient.send(msg)

class Audio(threading.Thread):
    def __init__(self, audio_buffer):
        threading.Thread.__init__(self)

        self.audio_buffer = audio_buffer

        # initialize portaudio
        self.audio_client = pyaudio.PyAudio()

    def run(self):
        stream = self.audio_client.open(format=pyaudio.paFloat32, channels=1, rate=SR, input=True, frames_per_buffer=CHUNKSIZE)

        # Do this as long as you want fresh samples
        data = stream.read(CHUNKSIZE, exception_on_overflow=False)

        samples_in_buffer = 0

        while data != '':
            numpydata = np.fromstring(data, dtype=np.float32)

            if samples_in_buffer < len(self.audio_buffer):
                # Continue to fill the audio buffer until 16s worth of audio is collected
                if samples_in_buffer + len(numpydata) > len(self.audio_buffer):
                    # Adding all of this data would make more than 16s
                    samples_extra = samples_in_buffer + len(numpydata) - len(self.audio_buffer)
                    self.audio_buffer = np.concatenate((self.audio_buffer[samples_extra:], numpydata))
                    samples_in_buffer = len(self.audio_buffer)
                else:
                    # Adding this data doesn't hit 16s (or is exactly 16s)
                    self.audio_buffer[len(self.audio_buffer) - (samples_in_buffer + len(numpydata)):len(self.audio_buffer) - samples_in_buffer] = numpydata
                    samples_in_buffer += len(numpydata)
            else:
                # Keep the most recent 16s
                self.audio_buffer = np.concatenate((self.audio_buffer[len(numpydata):], numpydata))

            data = stream.read(CHUNKSIZE, exception_on_overflow=False)

        # close stream
        stream.stop_stream()
        stream.close()
        self.audio_client.terminate()

class Predictor(threading.Thread):
    def __init__(self, model_name, arousal_model_filename, valence_model_filename, audio_buffer, predictions):
        threading.Thread.__init__(self)
        self.model_name = model_name
        self.audio_buffer = audio_buffer
        self.predictions = predictions

        split_name = model_name.split("_")
        self.model_type = split_name[0]

        # Load models
        self.arousal_model = pickle.load(open(self.model_type + "_models/" + arousal_model_filename, "rb"))
        self.valence_model = pickle.load(open(self.model_type + "_models/" + valence_model_filename, "rb"))

    def run(self):
        while True:
            if self.model_type == "mfcc":
                feature = calcMFCCs(self.audio_buffer)
            else:
                feature = calcConvnetFeatures(self.audio_buffer)

            valence = self.valence_model.predict(feature)
            arousal = self.arousal_model.predict(feature)

            self.predictions[self.model_name][0] = valence
            self.predictions[self.model_name][1] = arousal

Realtime("", "")