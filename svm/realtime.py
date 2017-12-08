import pyaudio
import OSC
import numpy as np
from util import *
import threading
from multiprocessing.dummy import Process
import os
import pickle
import time
import librosa
from easy_feature_extraction import extractFeatures, load_model

SR = 22050
CHUNKSIZE = SR/2

class Realtime():
    def __init__(self, final_model_aro, final_model_val):

        # Initalize an empty 16s buffer for audio
        self.audio_buffer = np.zeros(SR * 16)

        # Will contain predictions from each model
        self.predictions = {}

        # Initialize audio thread
        self.audio_thread = Audio(self.audio_buffer)

        # Initialize model threads
        # Initialize convnet models
        self.convnet_predictor_threads = []
        self.mfcc_predictor_threads = []

        self.arousal_selected_features = open('hypermodel_arousal_mse_delay/input_model_names.txt', 'rb').read().split(" ")
        self.valence_selected_features = open('hypermodel_valence_mse_delay/input_model_names.txt', 'rb').read().split(" ")

        all_feats = list(set(self.arousal_selected_features + self.valence_selected_features))

        for model_name in all_feats:
            print "loading", model_name
            feat_type = "mfcc" if model_name.startswith("mfcc") else "convnet"
            arousal_model_filename = 'MODEL_arousal_' + model_name + "_seconds.sav"
            valence_model_filename = 'MODEL_valence_' + model_name + "_seconds.sav"
            self.predictions[model_name] = [0, 0]
            if feat_type == "mfcc":
                self.mfcc_predictor_threads.append(Predictor(model_name, arousal_model_filename, valence_model_filename,
                                                        self.audio_buffer, self.predictions))
            else:
                self.convnet_predictor_threads.append(Predictor(model_name, arousal_model_filename, valence_model_filename,
                                                           self.audio_buffer, self.predictions))

        # Initialize final model
        self.final_arousal_model = pickle.load(open(final_model_aro, "rb"))
        self.final_valence_model = pickle.load(open(final_model_val, "rb"))

        self.audio_thread.start()
        [convnet_predictor_thread.start() for convnet_predictor_thread in self.convnet_predictor_threads]
        [mfcc_predictor_thread.start() for mfcc_predictor_thread in self.mfcc_predictor_threads]

        print "Done init"

    def run(self):
        # Initialize OSC
        guiServer = OSC.OSCServer(('127.0.0.1', 7100))
        guiServerThread = Process(target=guiServer.serve_forever)
        guiServerThread.daemon = True
        guiServerThread.start()

        guiClient = OSC.OSCClient()
        guiClient.connect(('127.0.0.1', 6000))

        print "I AM RUNNING"
        while True:
            # Predict and send to the front end
            final_valence_feature = []
            final_arousal_feature = []

            for model_name in self.predictions:
                if model_name in self.valence_selected_features:
                    prediction = self.predictions[model_name]
                    final_valence_feature.append(prediction[0])

                if model_name in self.arousal_selected_features:
                    prediction = self.predictions[model_name]
                    final_arousal_feature.append(prediction[1])

            valence = self.final_valence_model.predict([final_valence_feature])
            arousal = self.final_arousal_model.predict([final_arousal_feature])
            confidence = 1 - (0.5 * np.std(np.array(final_arousal_feature)) + 0.5 * np.std(np.array(final_valence_feature)))

            time.sleep(0.25)
            #print valence, arousal, confidence

            self.sendOSCMessage("/prediction", guiClient, [valence[0], arousal[0], confidence])


    def sendOSCMessage(self, addr, guiClient, *msgArgs):
        msg = OSC.OSCMessage()
        msg.setAddress(addr)
        msg.append(*msgArgs)
        guiClient.send(msg)

class Audio(Process):
    def __init__(self, audio_buffer):
        Process.__init__(self)

        self.audio_buffer = audio_buffer
        self.daemon = True

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
                    self.audio_buffer[:len(self.audio_buffer) - samples_extra] = self.audio_buffer[len(self.audio_buffer) - samples_in_buffer:]
                    self.audio_buffer[len(self.audio_buffer) - len(numpydata):] = numpydata
                    samples_in_buffer = len(self.audio_buffer)
                else:
                    # Adding this data doesn't hit 16s (or is exactly 16s)
                    self.audio_buffer[len(self.audio_buffer) - (samples_in_buffer + len(numpydata)):len(self.audio_buffer) - samples_in_buffer] = numpydata
                    samples_in_buffer += len(numpydata)
            else:
                # Keep the most recent 16s
                self.audio_buffer[:len(self.audio_buffer) - len(numpydata)] = self.audio_buffer[len(numpydata):]
                self.audio_buffer[len(self.audio_buffer) - len(numpydata):] = numpydata

            data = stream.read(CHUNKSIZE, exception_on_overflow=False)

        # close stream
        stream.stop_stream()
        stream.close()
        self.audio_client.terminate()

class Predictor(Process):
    def __init__(self, model_name, arousal_model_filename, valence_model_filename, audio_buffer, predictions):
        Process.__init__(self)
        self.model_name = model_name
        self.audio_buffer = audio_buffer
        self.predictions = predictions
        self.convnets = [load_model(mid_idx) for mid_idx in range(5)]
        self.daemon = True

        split_name = model_name.split("_")
        self.model_type = split_name[0]
        self.length = 0.5 if split_name[1] == "half" else float(split_name[1])

        # Load models
        self.arousal_model = pickle.load(open(self.model_type + "_models/" + arousal_model_filename, "rb"))
        self.valence_model = pickle.load(open(self.model_type + "_models/" + valence_model_filename, "rb"))

    def run(self):
        while True:
            samples_desired = int(self.length * SR)
            audio = np.array(self.audio_buffer[len(self.audio_buffer) - samples_desired:])
            audio = librosa.core.resample(np.array(audio), 22050, 12000)
            failed = False
            go = False

            if self.model_type == "mfcc":
                feature = calcMFCCs([audio])
            else:
                while np.shape(audio)[0] < 348000:
                    audio = np.concatenate((audio, audio), axis=0)
                feature = [np.array(extractFeatures(audio, self.convnets)).flatten()]
                print self.model_name

            valence = self.valence_model.predict(feature)
            arousal = self.arousal_model.predict(feature)

            self.predictions[self.model_name][0] = valence[0]
            self.predictions[self.model_name][1] = arousal[0]

main = Realtime("hypermodel_arousal_mse_delay/hypermodel.sav", "hypermodel_valence_mse_delay/hypermodel.sav")
print "Starting main"
main.run()