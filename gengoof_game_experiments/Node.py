import numpy as np
import random

# Implementation of Node Object
class Node:
	'''
	Represents a single node h in the set H of the game tree
	(or empirical game tree)
	
	@feature (int) player_id: identifies which player number in N
		acts at this decision node
	@feature (tuple) infoset_id: identifies which player's set of infosets
		(I_i) this infoset belongs to, and which infoset I \in I_i this node h
		belongs to; player_id should be the same as infoset_id[0]
	@feature (list) history: list of parent nodes from root to h
	@feature (list) children: list of child nodes, which must be added after h
		has been instantiated
	'''
	def __init__(self, player_id, infoset_id, history, action_space, N):

		self.num_players = N
		self.player_id = player_id
		self.infoset_id = infoset_id
		self.history = history
		self.children = []
		self.strategy = None
		self.is_terminal = False
		self.utility = None
		self.action_space = action_space
		self.is_chance = False
		self.use_beliefs = False
		self.belief = None
		if player_id == 0:
			self.is_chance = True

	def add_children(self, children):
		'''
		@arg (list)(Nodes) children: list of child nodes reached when the player
		at h take an action

		Adds child nodes and edges to h to expand the tree
		'''
		if self.children is None:
			self.children = children
		else:
			self.children += children

	def make_terminal(self, utility):
		'''
		@arg (N-long list) (floats) utility: payoff value
		
		Adds h to the set T of terminal nodes and gives it a payoff value
		'''
		self.utility = utility
		self.is_terminal = True
		self.player_id = None
		self.infoset_id = None
		self.action_space = None

	def make_copy(self):
		'''
		Returns an identical Node object, distinguished from this one
		'''
		nn = None
		if self.is_terminal:
			nn = Node(self.player_id, self.infoset_id, self.history[:], None, self.num_players)
			nn.children = None
		else:
			nn = Node(self.player_id, self.infoset_id, self.history[:], self.action_space[:], self.num_players)
			nn.children = self.children[:]
		nn.utility = self.utility
		nn.is_terminal = self.is_terminal
		nn.is_chance = self.is_chance

		return nn


	def get_child_given_action(self, action):
		'''
		@arg (tup or str) action: given player action in game

		Returns the child of the current Node object, given its chosen action to play
		'''
		assert action in self.action_space
		new_history = self.history + [action]

		def next_node_gen(children, new_history):
			for y in children:
				if y.history == new_history:
					yield y

		next_node = list(next_node_gen(self.children, new_history))
		if next_node == []:
			return None

		return next_node[0]

	def get_prob_dist_given_chance_map(self, chance_map):
		'''
		'''
		card_weights = chance_map.get(self).copy()
		prob_dist = {}
		denom = sum(card_weights.values())
		for e in card_weights.keys():
			prob_dist[e] = card_weights.get(e) / denom

		return prob_dist

	def compute_pay(self, strategy_profile, chance_map, input_reach_prob):
		'''
		@arg (map: Infoset --> (map: str --> float)) strategy_profile: each key
			in the outer map is a player infoset. Each player infoset's strategy is represented
			as a second map giving a distribution over that infoset's action space
		@arg (map: Node --> (map: str --> float)) chance_map: map from each chance node to the 
			probability distribution over the possible outcomes associated with that node
		@arg (float) input_reach_prob: probability of reaching this current node

		Compute the expected payoff for all players in the subgame rooted at this node,
		given a joint strategy profile 
		'''
		pay = None
		if self.is_terminal:
			return self.utility * input_reach_prob

		elif self.is_chance:
			pay = np.zeros(self.num_players)
			prob_dist = self.get_prob_dist_given_chance_map(chance_map)
			for outcome in prob_dist.keys():
				next_node = self.get_child_given_action(outcome)
				if next_node is not None:
					next_reach_prob = input_reach_prob * prob_dist.get(outcome)
					pay = pay + next_node.compute_pay(strategy_profile, chance_map, next_reach_prob)

		else:
			pay = np.zeros(self.num_players)
			infoset_strat = strategy_profile.get(self.infoset_id)
			
			if infoset_strat is not None:
				for a in infoset_strat.keys():
					next_node = self.get_child_given_action(a)
					if next_node is not None:
						next_reach_prob = input_reach_prob * infoset_strat.get(a, 0.0)
						
						if next_reach_prob > 0.0:
							pay = pay + next_node.compute_pay(strategy_profile, chance_map, next_reach_prob)

		return pay
