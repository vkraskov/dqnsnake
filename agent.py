# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, merge, concatenate, add, Input, Multiply, Merge
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Permute
from keras.layers.recurrent import LSTM
from keras.initializers import RandomUniform
from keras.layers import TimeDistributed

from game import ACT_FORWARD, ACT_BACK, ACT_RIGHT, ACT_LEFT

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

STATE_DXY = 16
STATE_SIZE = STATE_DXY*STATE_DXY
STATE_CLIP_DXY = STATE_DXY/4
HIST_MAXLEN = 15
HIST_FEATURES_SIZE = 1

action2signal = { ACT_FORWARD: [0.9, -0.3, -0.3, -0.3], ACT_BACK: [-0.3, 0.9, -0.3, -0.3], ACT_RIGHT: [-0.3, -0.3, 0.9, -0.3], ACT_LEFT: [-0.3, -0.3, -0.3, 0.9] }

############

class Agent:
	def __init__(self, maxsteps, action_size, dqnmem_size):
		self.maxsteps = maxsteps
		self.state_size = STATE_SIZE
		self.action_size = action_size
		self.dqnmem_size = dqnmem_size
		self.memory = deque(maxlen=dqnmem_size)
		self.memory_fail = deque(maxlen=dqnmem_size)
		self.memory_good = deque(maxlen=dqnmem_size)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.001
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self._init_hist_mem()
		self.model = self._build_model()
		self.mem_seq_id = 0

	def newgame(self):
		self._init_hist_mem()

	def _init_hist_mem(self):
		a = 0 
		#self.noaction = np.zeros(self.action_size, dtype=np.float)
		#self.hist_mem = deque(maxlen=HIST_MAXLEN)
		#for i in range(HIST_MAXLEN): 
		#	self.hist_mem.append(self.noaction)
		#	#self.hist_mem.append([-1., 0., 0.])

	def _build_model(self):
		# https://github.com/fchollet/keras/issues/1860
		# shape(out) = (shape(input) - kernelsize + 2*pad)/strides+1
		#
		# https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py

		# https://stackoverflow.com/questions/43152053/appending-layers-with-previous-in-keras-conv2d-object-has-no-attribute-is-p
		in1 = Input(shape=(STATE_DXY, STATE_DXY, 1))
		conv2d_1_1 = Conv2D(32, (8, 8), activation = 'relu')(in1)
		conv2d_1_2 = Conv2D(64, (4, 4), activation = 'relu')(conv2d_1_1)
		flatten_1 = Flatten()(conv2d_1_2)
		dense_1_1 = Dense(256, activation='relu')(flatten_1)

		in2 = Input(shape=(STATE_CLIP_DXY, STATE_CLIP_DXY, 1))
		conv2d_2_1 = Conv2D(16, (2, 2), activation = 'relu')(in2)
		conv2d_2_2 = Conv2D(32, (1, 1), activation = 'relu')(conv2d_2_1)
		flatten_2 = Flatten()(conv2d_2_2)
		dense_2_1 = Dense(256, activation='relu')(flatten_2)

		#in3 = Input(shape=(1,))
		#dense_3_1 = Dense(256, activation='relu')(in3)

		#joined = keras.layers.Merge()([dense_1_1, dense_2_1, dense_3_1])
		joined = keras.layers.Merge()([dense_1_1, dense_2_1])
		dense1 = Dense(256, activation='relu')(joined)
                dense2 = Dense(self.action_size, activation='linear')(dense1)

		#in3 = Input(shape=(1,))
		#dense_3_1 = Dense(256, activation='relu')(in3)
		#dense_3_2 = Dense(self.action_size, activation='relu')(dense_3_1)
		#
		#joined1 = keras.layers.Merge()([dense_1_1, dense_2_1])
		#dense1 = Dense(256, activation='relu')(joined1)
                #dense2 = Dense(self.action_size, activation='linear')(dense1)
		#joined2 = keras.layers.Merge()([dense2, dense_3_2])
		#
		#model = Model(inputs = [in1 , in2, in3], outputs = joined2)

		model = Model(inputs = [in1 , in2], outputs = dense2)
                model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
		model.summary()

        	return model

	def remember(self, state, action, reward, next_state, done, step, score):
		np_time = np.array([0 * (2.*step/float(self.maxsteps)-1.)])
		self.memory.append([state, action, reward, next_state, done, np_time, self.mem_seq_id])
		#print self.memory[0][5]
		#if done:
		#	prev_state  = self.get_state_byid(self.mem_seq_id-1)
		#	if prev_state != None:
		#		self.memory_fail.append(prev_state)
		#	self.memory_fail.append([state, action, reward, next_state, done, hist_mem, self.mem_seq_id])
		#if reward > 0:
		#	prev_state  = self.get_state_byid(self.mem_seq_id-1)
		#	if prev_state != None:
		#		self.memory_good.append(prev_state)
		#	self.memory_good.append([state, action, reward, next_state, done, hist_mem, self.mem_seq_id])
		self.mem_seq_id += 1

	def act(self, state, step, score):
		if self.epsilon_min < self.epsilon:
			if np.random.rand() <= self.epsilon:
				random_action = random.randrange(self.action_size)
			#	if random_action == ACT_BACK: 
			#		print "random_action == BACK"
				return random_action
		np_state = np.asarray(state).reshape(1, STATE_DXY, STATE_DXY, 1) 
		state_clip = self.get_state_clip(state, STATE_CLIP_DXY)
		np_state_clip =  np.asarray(state_clip).reshape(1, STATE_CLIP_DXY, STATE_CLIP_DXY, 1)
		np_time = np.array([0 * (2.*step/float(self.maxsteps)-1.)])
		np_time = np_time.reshape(1, 1)
		#print "np_time", np_time
		act_values = self.model.predict([np_state, np_state_clip])
		#act_values = self.model.predict([np_state, np_state_clip, np_time])
		#if np.argmax(act_values[0]) == ACT_BACK: 
		#	print act_values, np.argmax(act_values[0]), action2str[np.argmax(act_values[0])]
		return np.argmax(act_values[0])  # returns action

	def train_batch(self, X_batch, X_batch_hist, y_batch):
		X_batch_clip = self.get_state_clip_batch(X_batch, STATE_CLIP_DXY)
		#print "X_batch.shape", X_batch.shape
		#print "X_batch_clip.shape", X_batch_clip.shape
		#print "X_batch_hist.shape", X_batch_hist.shape
		return self.model.fit([X_batch, X_batch_clip], y_batch, epochs=1, verbose=0)
		#return self.model.fit([X_batch, X_batch_clip, X_batch_hist], y_batch, epochs=1, verbose=0)

	def predict_batch(self, X_batch, X_hist):
		X_batch_clip = self.get_state_clip_batch(X_batch, STATE_CLIP_DXY)
		return self.model.predict_on_batch([X_batch, X_batch_clip])
		#return self.model.predict_on_batch([X_batch, X_batch_clip, X_hist])

	def get_state_byid(self, mem_id):
		for i in range(len(self.memory)):
			if self.memory[i][5] == mem_id:
				return self.memory[i]
		for i in range(len(self.memory_fail)):
			if self.memory_fail[i][5] == mem_id:
				return self.memory_fail[i]
		for i in range(len(self.memory_good)):
			if self.memory_good[i][5] == mem_id:
				return self.memory_good[i]
		return None

	def create_batch(self, memory, batch_size):
		# https://gist.github.com/kkweon/5605f1dfd27eb9c0353de162247a7456
		sample = random.sample(memory, batch_size)
		sample = np.asarray(sample)

		s1 = sample[:, 0]
		a1 = sample[:, 1].astype(np.int8)
		r1 = sample[:, 2]
		s2 = sample[:, 3]
		d1 = sample[:, 4] * 1.
		h1 = sample[:, 5]
		i1 = sample[:, 6]

		X_batch = np.vstack(s1)
		X_batch = np.asarray(X_batch).reshape(batch_size, STATE_DXY, STATE_DXY, 1) 
		X_batch_hist = np.vstack(h1)
		X_batch_hist = np.asarray(X_batch_hist).reshape(batch_size, 1)
		y_batch = self.predict_batch(X_batch, X_batch_hist)

		X_batch_s2 = np.vstack(s2)
		X_batch_s2 = np.asarray(X_batch_s2).reshape(batch_size, STATE_DXY, STATE_DXY, 1) 

		X_batch_s3 = np.zeros(batch_size*STATE_DXY*STATE_DXY, dtype=np.float)
		X_batch_s3 = np.asarray(X_batch_s3).reshape(batch_size, STATE_DXY, STATE_DXY)
		r2 = np.zeros(batch_size, dtype=np.float)
		d2 = np.zeros(batch_size, dtype=np.float)
		for k in range(batch_size):
			k_id = i1[k]
			mem  = self.get_state_byid(k_id+1)
			#print "mem:", mem
			if mem != None:
				#print "X_batch_s3::mem", mem[2], mem[4] * 1.
				X_batch_s3[k] = mem[3]
				d2[k] = mem[4] * 1.
				r2[k] = mem[2]
				#h2[k] = mem[5]
			else:
				#print "X_batch_s3::mem == None"
				d2[k] = 1 * 1.
				r2[k] = -100.0
		X_batch_s3 = np.asarray(X_batch_s3).reshape(batch_size, STATE_DXY, STATE_DXY, 1)

		X_batch_s4 = np.zeros(batch_size*STATE_DXY*STATE_DXY, dtype=np.float)
		X_batch_s4 = np.asarray(X_batch_s4).reshape(batch_size, STATE_DXY, STATE_DXY)
		r3 = np.zeros(batch_size, dtype=np.float)
		d3 = np.zeros(batch_size, dtype=np.float)
		h3 = np.zeros(shape=(batch_size, 1), dtype=np.float)
		#h3 = np.zeros(shape=(batch_size, HIST_MAXLEN, HIST_FEATURES_SIZE), dtype=np.float)
		for k in range(batch_size):
			k_id = i1[k]
			mem  = self.get_state_byid(k_id+2)
			#print "mem:", mem
			if mem != None:
				#print "X_batch_s4::mem", mem[2], mem[4] * 1.
				X_batch_s4[k] = mem[3]
				d3[k] = mem[4] * 1.
				r3[k] = mem[2]
				h3[k] = mem[5]
				print "h3[k]", h3[k]
			else:
				#print "X_batch_s4::mem == None"
				d3[k] = 1 * 1.
				r3[k] = -100.0
				for i in range(HIST_MAXLEN):
					h3[k] = [0*-1.0]
					#h3[k][i] = [-1.]
					#h3[k][i] = [-1., 0., 0.]
		X_batch_s4 = np.asarray(X_batch_s4).reshape(batch_size, STATE_DXY, STATE_DXY, 1)
		X_batch_s4_hist = np.asarray(h3).reshape(batch_size, 1)
		#print "X_batch_s4.shape", X_batch_s4.shape
		#print "X_batch_s4_hist.shape", X_batch_s4_hist.shape, X_batch_s4_hist
		#X_batch_s4_hist = np.asarray(X_batch_s4_hist).reshape(batch_size, HIST_MAXLEN, HIST_FEATURES_SIZE)
		#print "X_batch_s3:", X_batch_s3[0]

		#print r1[0], self.gamma * (np.max(self.predict_batch(X_batch_s3), 1) * (1 - d1))[0] 
		#y_batch[np.arange(batch_size), a1] = r1 + self.gamma * np.max(self.predict_batch(X_batch_s2), 1) * (1 - d1)
		#print r1[0], self.gamma*r2[0], self.gamma*r2[0]*(1 - d1[0])*(1 - d2[0]),  self.gamma * self.gamma * (np.max(self.predict_batch(X_batch_s3), 1) * (1 - d2) * (1 - d1))[0] 
		y_batch[np.arange(batch_size), a1] = \
				r1 + \
				self.gamma * r2 * (1-d1)*(1-d2)*(1-d3) + \
				self.gamma * self.gamma * r3 * (1-d2)*(1-d1)*(1-d3) + \
				self.gamma * self.gamma * self.gamma * np.max(self.predict_batch(X_batch_s4, X_batch_s4_hist), 1) * (1-d2) * (1-d1)*(1-d3)


		#print "X_batch_hist", X_batch_hist
		return X_batch, X_batch_hist, y_batch

	def get_state_clip(self, state, size):
		beg_x = int( (len(state) - size)/2 )
		beg_y = int( (len(state[0]) - size)/2 )
		end_x = beg_x + size-1
		end_y = beg_y + size-1
		#
		clip = np.zeros(size*size, dtype=np.float)
		clip = np.asarray(clip).reshape(size, size)
		#
		for y in range(beg_y, end_y+1):
			for x in range(beg_x, end_x+1):
				clip[x-beg_x][y-beg_y] = state[x][y]
		return clip

	def get_state_clip_batch(self, batch, size):
		batch_tmp = np.asarray(batch).reshape(len(batch), STATE_DXY, STATE_DXY)
		batch_clip = np.zeros(len(batch)*size*size, dtype=np.float)
		batch_clip = np.asarray(batch_clip).reshape(len(batch), size, size)
		for i in range(len(batch)):
			batch_clip[i] = self.get_state_clip(batch_tmp[i], size)
		batch_clip = np.asarray(batch_clip).reshape(len(batch), size, size, 1)
		return batch_clip

	def replay(self, batch_size):
		#print "replay from memory random.."
		X_batch, X_batch_hist, y_batch = self.create_batch(self.memory, batch_size)
		self.train_batch(X_batch, X_batch_hist, y_batch)

		#print "replay from memory fails & wins.."
		#if len(self.memory_fail)>0:
		#	batch_fail_size = min(16, len(self.memory_fail))
		#	X_batch, y_batch = self.create_batch(self.memory_fail, batch_fail_size)
		#	self.train_batch(X_batch, y_batch)

		#if len(self.memory_good)>0:
		#	batch_good_size = min(16, len(self.memory_good))
		#	X_batch, y_batch = self.create_batch(self.memory_good, batch_good_size)
		#	self.train_batch(X_batch, y_batch)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	#	tmp_mem = []
	#	for e in self.memory:
	#		if done:
	#			tmp_mem.append([e.state, e.action, e.reward, e.next_state, e.done])
	#	minibatch = random.sample(tmp_mem, min(1000, len(tmp_mem)))
	#	for state, action, reward, next_state, done in minibatch:
	#		target = reward
	#		np_state = np.asarray(state).reshape(1, STATE_DXY, STATE_DXY, 1) 
	#		target_f = self.model.predict(np_state)
	#		target_f[0][action] = target
	#		self.model.fit(np_state, target_f, epochs=1, verbose=0)
	#	#
	#	minibatch = random.sample(self.memory, batch_size)
	#	for state, action, reward, next_state, done in minibatch:
	#	    target = reward
	#	    if not done:
	#		np_next_state = np.asarray(next_state).reshape(1, STATE_DXY, STATE_DXY, 1)
	#		target = (reward + self.gamma * np.amax(self.model.predict(np_next_state)[0]))
	#	    np_state = np.asarray(state).reshape(1, STATE_DXY, STATE_DXY, 1) 
	#	    target_f = self.model.predict(np_state)
	#	    target_f[0][action] = target
	#	    self.model.fit(np_state, target_f, epochs=1, verbose=0)
	#	if self.epsilon > self.epsilon_min:
	#	    self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

