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
from keras.initializers import RandomUniform

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))

STATE_DXY = 16
STATE_SIZE = STATE_DXY*STATE_DXY
STATE_CLIP_DXY = STATE_DXY/4
STATE_NEAR_DXY = 3

############

class Agent:
	def __init__(self, action_size, maxsteps):
		self.state_size = STATE_SIZE
		self.maxsteps = maxsteps
		self.action_size = action_size
		self.sdict = {}
		self.memory = deque(maxlen=self.maxsteps*4)
		self.memory_fail = deque(maxlen=self.maxsteps)
		self.memory_good = deque(maxlen=self.maxsteps)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.001
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()
		self.mem_seq_id = 0
		self.act_values = None

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

		in3 = Input(shape=(STATE_NEAR_DXY, STATE_NEAR_DXY, 1))
		conv2d_3_1 = Conv2D(16, (3, 3), activation = 'relu')(in3)
		conv2d_3_2 = Conv2D(32, (1, 1), activation = 'relu')(conv2d_3_1)
		flatten_3 = Flatten()(conv2d_3_2)
		dense_3_1 = Dense(256, activation='relu')(flatten_3)

		in4 = Input(shape=(1,))
		dense_4_1 = Dense(256, activation='relu')(in4)

		joined = keras.layers.Merge()([dense_1_1, dense_2_1, dense_3_1, dense_4_1])
		dense_f_1 = Dense(256, activation='relu')(joined)
		dense_f_2 = Dense(self.action_size, activation='linear')(dense_f_1)

		model = Model(inputs = [in1 , in2, in3, in4], outputs = dense_f_2)
                model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
		model.summary()

        	return model

	def remember(self, state, action, reward, next_state, done, step, score):
		nn_step = (2.*float(score)/self.maxsteps-1)
		rec = [state, action, reward, next_state, done, nn_step, score, self.mem_seq_id]
		self.sdict[self.mem_seq_id] = rec
		# removing records older mem fail mem size multiplied by maxsteps in episode
		delidx = self.mem_seq_id-(self.maxsteps*self.maxsteps)
		if delidx >= 0:
			del self.sdict[delidx]
		self.memory.append(rec)
		if done:
			self.memory_fail.append(rec)
		if reward > 0:
			self.memory_good.append(rec)
		self.mem_seq_id += 1

	def act(self, state, step, score):
		if self.epsilon_min < self.epsilon:
			if np.random.rand() <= self.epsilon:
				random_action = random.randrange(self.action_size)
			#	if random_action == ACT_BACK: 
			#		print "random_action == BACK"
				return random_action
		# wide view
		np_state = np.asarray(state).reshape(1, STATE_DXY, STATE_DXY, 1) 
		# clipped view
		state_clip = self.get_state_clip(state, STATE_CLIP_DXY)
		np_state_clip =  np.asarray(state_clip).reshape(1, STATE_CLIP_DXY, STATE_CLIP_DXY, 1)
		# near view
		state_near = self.get_state_clip(state, STATE_NEAR_DXY)
		np_state_near =  np.asarray(state_near).reshape(1, STATE_NEAR_DXY, STATE_NEAR_DXY, 1)
		# score as tails grows
		nn_step = (2.*float(score)/self.maxsteps-1)
		np_step_arr =  np.asarray(nn_step).reshape(1, 1)
		# predict
		self.act_values = self.model.predict([np_state, np_state_clip, np_state_near, np_step_arr])
		#if np.argmax(act_values[0]) == ACT_BACK: 
		#	print act_values, np.argmax(act_values[0]), action2str[np.argmax(act_values[0])]
		return np.argmax(self.act_values[0])  # returns action

	def train_batch(self, X_batch, X_extra, y_batch):
		X_batch_clip = self.get_state_clip_batch(X_batch, STATE_CLIP_DXY)
		X_batch_near = self.get_state_clip_batch(X_batch, STATE_NEAR_DXY)
		return self.model.fit([X_batch, X_batch_clip, X_batch_near, X_extra], y_batch, epochs=1, verbose=0)

	def predict_batch(self, X_batch, X_extra):
		X_batch_clip = self.get_state_clip_batch(X_batch, STATE_CLIP_DXY)
		X_batch_near = self.get_state_clip_batch(X_batch, STATE_NEAR_DXY)
		return self.model.predict_on_batch([X_batch, X_batch_clip, X_batch_near, X_extra])

	def get_state_byid(self, mem_id):
		return self.sdict[mem_id]

	def create_batch(self, memory, batch_size):
		# https://gist.github.com/kkweon/5605f1dfd27eb9c0353de162247a7456
		sample = random.sample(memory, batch_size)
		sample = np.asarray(sample)

		s1 = sample[:, 0]
		a1 = sample[:, 1].astype(np.int8)
		r1 = sample[:, 2]
		s2 = sample[:, 3]
		d1 = sample[:, 4] * 1.
		t1 = sample[:, 5]
		w1 = sample[:, 6]
		i1 = sample[:, 7]

		X_batch = np.vstack(s1)
		X_extra = np.vstack(w1)
		X_batch = np.asarray(X_batch).reshape(batch_size, STATE_DXY, STATE_DXY, 1) 
		y_batch = self.predict_batch(X_batch, X_extra)

		#X_batch_s2 = np.vstack(s2)
		#X_batch_s2 = np.asarray(X_batch_s2).reshape(batch_size, STATE_DXY, STATE_DXY, 1) 
		#X_batch_s3 = np.zeros(batch_size*STATE_DXY*STATE_DXY, dtype=np.float)
		#X_batch_s3 = np.asarray(X_batch_s3).reshape(batch_size, STATE_DXY, STATE_DXY)

		X_batch_s4 = np.zeros(batch_size*STATE_DXY*STATE_DXY, dtype=np.float)
		X_batch_s4 = np.asarray(X_batch_s4).reshape(batch_size, STATE_DXY, STATE_DXY)
		rx = np.zeros(batch_size, dtype=np.float)
		t3 = np.zeros(batch_size, dtype=np.float)
		#dx = np.zeros(batch_size, dtype=np.float)
		for k in range(batch_size):
			edr = 0.0 # sum of discounted rewards
			k_id = i1[k]
			for z in range(7):
				try:
					mem  = self.get_state_byid(k_id+z)
				except KeyError, e:
					print 'I got a KeyError - reason "%s"' % str(e)
					mem = None
				if mem == None:
					break
				t3[k] = mem[6]
				X_batch_s4[k] = mem[3]
				edr = edr + mem[2]*self.gamma**z
				if mem[4] == 1: 
					edr = -1.0
					break
			#print "edr", edr
			rx[k] = edr

		X_batch_s4 = np.asarray(X_batch_s4).reshape(batch_size, STATE_DXY, STATE_DXY, 1)
		X_extra_s4 = np.vstack(t3)
		X_extra_s4 = np.asarray(X_extra_s4).reshape(batch_size, 1)
		y_batch[np.arange(batch_size), a1] = rx + np.max(self.predict_batch(X_batch_s4, X_extra_s4), 1)*self.gamma**7

		return X_batch, X_extra, y_batch

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
		X_batch, X_extra, y_batch = self.create_batch(self.memory, batch_size)
		self.train_batch(X_batch, X_extra, y_batch)

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

