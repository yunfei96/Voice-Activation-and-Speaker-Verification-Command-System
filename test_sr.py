import speech_recognition as sr

from scipy.io import wavfile
import numpy as np



fs, data = wavfile.read("test.wav")
#m = max(data)
data = data*32767
data = data.astype(np.int16)
#wavfile.write("new.wav",fs,np.asarray(data))

# obtain audio from the microphone
r = sr.Recognizer()
r.grammars = ["release", "grasp"];
audio = sr.AudioData(data.tobytes(), fs, 2)

# recognize speech using Sphinx
try:
    print("Google thinks you said " + r.recognize_google(audio, key = None, language = "en-US", show_all = False))
except sr.UnknownValueError:
    print("Google could not understand audio")
except sr.RequestError as e:
    print("Google error; {0}".format(e))
