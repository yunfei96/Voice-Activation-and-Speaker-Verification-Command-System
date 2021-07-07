'''

this program will check the loudness of each 0.5 second
if the loudness is larger than threshold, the 0.5 second will be append to a sentence
if there is 0.5 second margin, the sentence will be feed into a text detector

'''
import mbn as mbn
from scipy import spatial
import numpy as np
import tensorflow as tf
import mobilenet
import pyaudio
from scipy.io import wavfile
from queue import Queue
import os
import speech_recognition as sr
import filter
import subprocess
import time

def distance(list1, list2):
    """Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    return sum(squares) ** .5

#-----------------------------set up OS-------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#------------------------set up deep learning network-----------------------------
print("Loading model weights")
mn_model = mobilenet.MobileNet((512, 300, 1), 1251, alpha=0.75, include_top=True, weights=None)
mn_model.load_weights('/Users/yunfeiguo/PycharmProjects/loudness_detection/model/mbn_model.h5')
m = tf.keras.Model(mn_model.input, mn_model.layers[-2].output)
print(m.summary())

print("finish loading deep learning model")


#enroll 1 command
enroll = mbn.get_embedding(m, "/Users/yunfeiguo/Documents/research/rml_exoskeleton_dataset/wenda/grasp-cup/cup4.wav", 3)
#np.save("grasp", enroll)

#test distance between commands
test = mbn.get_embedding(m, "/Users/yunfeiguo/Documents/research/rml_exoskeleton_dataset/wenda/grasp-cup/cup2.wav", 3)
distances = spatial.distance.cosine(enroll, test)
#distances = distance(enroll, test)
print(distances)



# #fs, data = wavfile.read("/Users/yunfeiguo/Desktop/rml_exoskeleton_dataset/wenda/grasp-cup/cup4.wav")
# fs, data = wavfile.read("test.wav")
# # data = data*32767
# # data = data.astype(np.int16)
# # wavfile.write("new.wav",fs,np.asarray(data))
#
# # obtain audio from the microphone
# r = sr.Recognizer()
# audio = sr.AudioData(data,fs,1)
# try:
#     print("Google thinks you said " + r.recognize_google(audio, key = None, language = "en-US", show_all = False))
# except sr.UnknownValueError:
#     print("Google could not understand audio")
# except sr.RequestError as e:
#     print("Google error; {0}".format(e))