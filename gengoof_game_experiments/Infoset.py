# Implementation of the Infoset object
import numpy as np
class Infoset:

	def __init__(self, infoset_id, node_list, action_space, N):
		'''
		'''
		for n in node_list:
			assert n.infoset_id == infoset_id

		self.infoset_id = infoset_id
		self.node_list = node_list
		self.action_space = action_space
		self.strategy = None
		self.regret_sum = None
		self.strategy_sum = None
		self.reach_prob_sum = None
		self.reach_prob = None
		self.action_utils = None
		self.num_players = N

	def set_strategy(self, strategy):
		'''
		Assign a strategy to the current information set
		'''
		self.strategy = strategy

	def compute_strategy(self):
		'''
		Update the strategy using cumulative regrets
		'''
		n_actions = len(self.action_space)
		make_positive = lambda x: np.where(x > 0, x, 0)
		remove_mistakes = lambda x: np.where(x < 10e-2, 0, x)
		strategy = remove_mistakes(make_positive(self.regret_sum))
		
		if np.array_equal(strategy, np.zeros(n_actions)):
			return np.repeat(1.0 / n_actions, n_actions)

		return strategy / sum(strategy)

	def update_strategy(self):
		'''
		Update the relative reach probabilities for the next step of 
		updating the strategy
		'''
		self.strategy_sum += self.reach_prob * self.strategy
		self.strategy = self.compute_strategy()
		self.reach_prob_sum += self.reach_prob
		self.reach_prob = 0.0

	def compute_average_strategy(self):
		'''
		Computes average strategy after T iterations of CFR algorithm.
		This will be the approx Nash equil. that CFR returns
		'''
		remove_mistakes = lambda x: np.where(x < 10e-3, 0, x)
		strategy = np.array([1.0 / len(self.action_space) for x in self.action_space])
		
		if self.reach_prob_sum > 0:
			strategy = remove_mistakes(self.strategy_sum / self.reach_prob_sum)

		return strategy / sum(strategy)

	def update_action_space(self, new_action):
		'''
		'''
		self.action_space.append(new_action)
		for n in self.node_list:
			cur_n = n.action_space[:]
			cur_n.append(new_action)
			n.action_space = cur_n

