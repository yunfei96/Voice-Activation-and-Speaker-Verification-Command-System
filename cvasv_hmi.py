'''

this program will check the loudness of each 0.5 second
if the loudness is larger than threshold, the 0.5 second will be append to a sentence
if there is 0.5 second margin, the sentence will be feed into a text detector

'''
import mbn as mbn
from scipy import spatial
import mobilenet
import numpy as np
import tensorflow as tf
import pyaudio
from scipy.io import wavfile
from queue import Queue
import os
import speech_recognition as sr
import filter
import subprocess
from tuning import release_list, grasp_list,active_list
import time
#-----------------------------set up OS-------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
r = sr.Recognizer()
r.grammars = ["release", "grasp", "hey glove"];
#------------------------set up deep learning network-----------------------------
print("Loading model weights")
print("Loading model weights")
mn_model = mobilenet.MobileNet((512, 300, 1), 1251, alpha=0.75, include_top=True, weights=None)
mn_model.load_weights('/Users/yunfeiguo/PycharmProjects/loudness_detection/model/mbn_model.h5')
m = tf.keras.Model(mn_model.input, mn_model.layers[-2].output)
#mbn.get_embedding(m, "/Users/yunfeiguo/Desktop/rml_exoskeleton_dataset/wenda/grasp-cup/cup4.wav", 3)
print("finish loading deep learning model")
datae = np.load("heyglove.npy");

# ----------------------- config py audio streaming ------------------------------
chunk_duration = 0.5  # Each read length in seconds from mic.
fs = 44100  # sampling rate for mic
chunk_samples = int(fs * chunk_duration)  # Each read length in number of samples.


# Queue to communicate between the audio callback and main thread
# ---------------------- detection threshold --------------------------------------
silence_threshold = 5000
q = Queue()
start = False
end = False
data = np.zeros(22050, dtype='int16')


def say(text):
    subprocess.call(['say', text])

def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream


def callback(in_data, frame_count, time_info, status):
    global data, silence_threshold, start, end
    data0 = np.frombuffer(in_data, dtype='int16')

    loudness = filter.get_freq(data0)


    #print(np.abs(loudness).mean())
    # smaller than threshold, ignore
    if np.abs(loudness).mean() < silence_threshold:
        # print('-', end = '', flush=True)
        if not start and not end:
            start = True
            data = np.zeros(22050, dtype='int16')
        elif start and end:
            q.put(data)
            #print("s", flush=True)
            start = False
            end = False
            #data = np.zeros(22050, dtype='int16')

        return (in_data, pyaudio.paContinue)
    else:
        # print('.', end = '',flush=True)
        if start:
            data = np.append(data, data0)
            end = True
        # data = np.append(data, data0)
        # q.put(data0)
    return (in_data, pyaudio.paContinue)


stream = get_audio_input_stream(callback)
stream.start_stream()

active = False

try:
    while 1:
        #start_time = time.clock()
        data = q.get()
        data = filter.output_freq(data)
        wavfile.write("test.wav", 44100, data)

        #fs, data = wavfile.read("test.wav")
        data = data * 32767
        data = data.astype(np.int16)
        # print(data.shape, flush=True)
        # print("d", flush=True)
        audio = sr.AudioData(data.tobytes(), fs, 2)
        #print(time.clock() - start_time, "seconds")
        command = ""
        try:
            command = r.recognize_google(audio, key=None, language="en-US", show_all=False)
            #print("command: " + command, flush=True)

        except sr.UnknownValueError:
            print("", flush=True)

        if not active:
            if command in active_list:
                print("Input command is: hey glove")
                datav = mbn.get_embedding(m, "test.wav", 3)
                distances = spatial.distance.cosine(datae, datav)

                print(distances)
                if distances < 0.4:
                    print("Hello, system is activated")
                    #say("Hello, yunfei")
                    active = True
        else:
            print(command)
            if command in grasp_list:
                print("the glove will grasp!")
                #say("grasp")
                #subprocess.call('echo guoyunfei | sudo -S echo 0,40,3,4,4,3,5> /dev/cu.usbmodem88249901',shell=True,  stdin=subprocess.PIPE)
            if  command in release_list:
                print("the glove will release!")
                #say("release")
                #subprocess.call('echo guoyunfei | sudo -S echo 0,0,0,0,0,0,0, > /dev/cu.usbmodem88249901',shell=True,  stdin=subprocess.PIPE)
            #else:
                #print("re-input command")




except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
