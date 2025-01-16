import numpy as np
import random
import tensorflow as tf
import time
# tf.compat.v1.disable_eager_execution()

from collections import deque
BATCH_SIZE = 64

class DQN:
	'''
	'''
	def __init__(self, state_shape, action_space, hp_set):
		self.state_shape = state_shape
		self.action_space = action_space
		self.memory = deque(maxlen=200000)
		self.epsilon = 1.0
		self.epsilon_decay = 0.995
		self.training_steps = hp_set["training_steps"]
		self.epsilon_min = hp_set["epsilon_min"]
		self.epsilon_annealing = hp_set["epsilon_annealing"]
		self.learning_rate = hp_set["learning_rate"]
		self.model_width = hp_set["model_width"]
		self.update_target = hp_set["update_target"]
		self.batch_size = BATCH_SIZE
		self.min_buffer_size = 1e4

		tf.compat.v1.disable_eager_execution()
		self.model = self.create_model()
		self.target_model = self.create_model()

	def create_model(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(self.model_width, input_shape=(self.state_shape[0],), activation="relu"))
		model.add(tf.keras.layers.Dense(self.model_width, activation="relu"))
		model.add(tf.keras.layers.Dense(len(self.action_space)))
		model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

		return model

	def remember(self, state, action, reward, next_state, done):
		'''
		'''
		self.memory.append([state, action, reward, next_state, done])

	def replay(self, step_num):
		'''
		'''
		if len(self.memory) < self.min_buffer_size:
			return 0
		
		samples = random.sample(self.memory, self.batch_size)
		current_states = np.array([sample[0][0] for sample in samples])
		current_qs_list = self.model.predict(current_states, batch_size=self.batch_size)
		next_states = np.array([sample[3][0] for sample in samples])
		future_qs_list = self.target_model.predict(next_states, batch_size=self.batch_size)

		X = []
		Y = []
		max_future_q_list = []

		for index, sample in enumerate(samples):
			state, action, reward, next_state, done = sample
			X.append(state[0])
			action_index = self.action_space.index(action)
			current_qs = current_qs_list[index]
			max_future_q = None
			if done:
				max_future_q = reward
			else:
				max_future_q = reward + np.max(future_qs_list[index])

			max_future_q_list.append(max_future_q)
			current_qs[action_index] = (1 - self.learning_rate) * current_qs[action_index] + self.learning_rate * max_future_q
			Y.append(current_qs)

		self.model.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)

		if step_num % self.update_target == 0:
			self.target_train()

		return np.mean(max_future_q_list)

	def target_train(self):
		'''
		'''
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i]
		self.target_model.set_weights(target_weights)

	def decrement_epsilon(self, steps):
		'''
		new method for updating epsilon with linear or exponential decay
		'''
		new_epsilon = None
		if self.epsilon_annealing == "linear":
			step_frac = 1.0 * (self.training_steps - steps) / self.training_steps
			new_epsilon = (self.epsilon - self.epsilon_min) * step_frac + self.epsilon_min
		else:
			assert self.epsilon_annealing == "exp"
			new_epsilon = self.epsilon * self.epsilon_decay

		return new_epsilon

	def act(self, state, is_game_start, steps):
		'''
		'''
		if self.epsilon > 0.0:
			self.epsilon = self.decrement_epsilon(steps)

		self.epsilon = max(self.epsilon_min, self.epsilon)
		actions = self.action_space[:]
		if is_game_start:
			actions = [x for x in self.action_space if x[0] not in [("deal",), ("walk",)]]

		if np.random.random() < self.epsilon:
			return random.choice(actions)

		q_output = self.model.predict(state)[0]
		index = np.argmax(q_output)
		if is_game_start and index >= len(actions):
			lenq = len(q_output)
			q_output_copy = q_output[:lenq-4]
			index = np.argmax(q_output_copy)
		
		return actions[index]

	def act_in_eval(self, state, is_game_start):
		'''
		'''
		actions = self.action_space[:]
		if is_game_start:
			actions = [x for x in self.action_space if x[0] not in [("deal",), ("walk",)]]

		q_output = self.model.predict(state)[0]
		index = np.argmax(q_output)
		if is_game_start and index >= len(actions):
			lenq = len(q_output)
			q_output_copy = q_output[:lenq-4]
			index = np.argmax(q_output_copy)

		return actions[index]

	def save_model(self, fn):
		'''
		'''
		self.model.save(fn)

