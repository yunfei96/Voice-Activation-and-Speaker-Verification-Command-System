import numpy as np
import  tensorflow as tf
import constants as c
from wav_reader import get_fft_spectrum
from scipy import spatial
import time
def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step)
	end_frame = int(max_sec*frames_per_sec)
	step_frame = int(step_sec*frames_per_sec)
	for i in range(0, end_frame+1, step_frame):
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv1
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets



def get_embedding(model, wav_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	signal = get_fft_spectrum(wav_file, buckets)
	#signal = np.dstack((signal, signal, signal))
	#embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape)))
	#s = time.clock()
	embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape,1)))
	#e = time.clock()
	#print(e-s)
	return embedding


def enroll_data(enroll):
	print("Loading model weights")
	mn_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(512, 300, 3), alpha=1, include_top=False, weights='imagenet', input_tensor=None, pooling='avg')
	new_layer2 = tf.keras.layers.Dense(1251, activation='softmax', name='final_layer')

	inp2 = mn_model.input
	out2 = new_layer2(mn_model.output)
	m = tf.keras.Model(inp2, out2)
	m.load_weights('/Users/yunfeiguo/PycharmProjects/loudness_detection/model/mbn_large_model.h5')
	m = tf.keras.Model(m.input, m.layers[-2].output)

	m.summary()
	print("Processing samples")
	data = get_embedding(m, enroll, 3)
	for i in range(10):
		data = data + get_embedding(m, enroll, 3)

	data = data/10.0
	np.save("data", data)
	return data

def verify_data(verify):
	print("Loading model weights")
	mn_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(512, 300, 3), alpha=1, include_top=False, weights='imagenet', input_tensor=None, pooling='avg')
	new_layer2 = tf.keras.layers.Dense(1251, activation='softmax', name='final_layer')

	inp2 = mn_model.input
	out2 = new_layer2(mn_model.output)
	m = tf.keras.Model(inp2, out2)
	m.load_weights('/Users/yunfeiguo/PycharmProjects/loudness_detection/model/mbn_large_model.h5')
	m = tf.keras.Model(m.input, m.layers[-2].output)

	m.summary()
	print("Processing samples")
	data = get_embedding(m, verify, 3)
	return data


if __name__ == '__main__':
	#enroll_data("/Users/yunfeiguo/PycharmProjects/loudness_detection/model/model.wav")
	datae = np.load("data.npy")
	datav = verify_data("/Users/yunfeiguo/PycharmProjects/loudness_detection/model/00001.wav")
	distances = spatial.distance.cosine(datae, datav)
	print(distances)
