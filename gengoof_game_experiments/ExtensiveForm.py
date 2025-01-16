import itertools as it
import gc
from Node import *
from Infoset import *
import math
from anytree import AnyNode, RenderTree
import networkx as nx

import mmappickle as mmp

ALPHA = 0.1

# Implementation of Extensive Form object
class ExtensiveForm:
	'''

	'''
	def __init__(self, infosets, root_node, terminal_nodes, chance_map, num_rounds):
		'''
		@arg (list of lists of Infosets) infosets: Collection I of information sets for each player, represented
			as a list of each player's Infoset objects, 1 through N
		@arg (list of Nodes) terminal_nodes: Collection T of terminal_nodes, represented as a list of Node
			objects
		@arg (list of dicts) strategy_space: List of initial pure strategies for each player, at each infoset;
			each dictionary (one per player minus Nature) in the list maps a player's infoset object to an action
			in the infoset's action space
		@arg (dict) chance_map: Map of each chance Node object h to the set of actions available to Nature
			at node h X(h)
		'''
		self.infosets = infosets
		self.terminal_nodes = terminal_nodes
		self.root = root_node
		self.chance_map = chance_map
		self.num_players = len(self.infosets)
		self.num_rounds = num_rounds

	def get_infoset_given_node(self, node):
		'''
		@arg (Node) node: given node in the game tree

		Helper method to find the infoset that contains the input node
		'''
		matching_infosets = [x for x in self.infosets[node.player_id - 1] if x.infoset_id == node.infoset_id]
		if len(matching_infosets) == 0:
			return None
		return matching_infosets[0]

	def compute_pay(self, strategy_profile):
		'''
		@arg (map: Infoset --> (map: str --> float)) strategy_profile: each key
			in the outer map is a player infoset. Each player infoset's strategy is represented
			as a second map giving a distribution over that infoset's action space

		Compute the expected payoff for all players, given a joint strategy profile
		'''
		return self.root.compute_pay(strategy_profile, self.chance_map, 1.0)

	def compute_reach_prob(self, strategy_profile, node):
		'''
		@arg (list of list of infoset, strategy map pairs) strategy_profile: each elt
			in the list is a player's strategy. Each player's strategy is represented
			as a list of tuples: (infoset, map giving a distribution over that infoset's
			action space)
		
		@arg (Node) node: Node object

		Compute the reach probability for a given node in the tree
		'''
		reach_prob = 1.0
		current = self.root

		for h in node.history:

			if current.player_id == 0:
				event_map = self.get_prob_dist_given_chance_map(current)
				if event_map.get(h) == 0.0:
					return 0.0
				reach_prob *= event_map.get(h)

			else:
				j = current.player_id - 1
				matching = [infoset_id for infoset_id in strategy_profile.keys() if infoset_id == current.infoset_id]
				if matching == []:
					matching_tups = [tup for tup in strategy_profile.keys() if current.infoset_id in tup]
					if len(matching_tups) == 0:
						return 0.0

					matching_tup = matching_tups[0]
					k = matching_tup.index(current.infoset_id)
					current_strategy = strategy_profile.get(matching_tup)
					action = [x for x in current_strategy.keys() if x[k] == h]

					if action == []:
						return 0.0

					action_ = action[0]
					reach_prob *= current_strategy.get(action_, 0.0)
				else:
					current_infoset_id = matching[0]
					current_strategy = strategy_profile.get(current_infoset_id)
					if current_strategy.get(h, 0.0) == 0.0:
						return 0.0
					
					reach_prob *= current_strategy.get(h, 0.0)
					if current.use_beliefs:
						reach_prob *= current.belief

			next_node = current.get_child_given_action(h)
			current = next_node

		return reach_prob

	def get_next_infoset_id(self, path, i):
		'''
		'''
		if len(path[i+1]) == 1:
			return 0

		num_player_actions = len([x for x in path[:(i+1)] if len(x) > 1])
		next_player_id = num_player_actions % 2 + 1
		next_infoset_id = None
		
		if next_player_id == 1:
			empir_history = tuple(path)[:(i+1)]
			next_infoset_id = (1, empir_history)
		else:
			empir_history = tuple(path)[:i]
			next_infoset_id = (2, empir_history)

		return next_infoset_id

	def update_game_with_simulation_output(self, observations, payoffs):
		'''
		Helper method intended for the EMPIRICAL game. Update the empirical game with new info
		from the simulator resulting from simulating a given strategy profile. This is how we 
		add brand new untraveled paths to the game tree

		This is how we
		update the empirical leaf utilities and Nature's empirical probability distributions.
		'''
		# if it's the first iteration, assume we at least have the root node defined, but w/o an 
		# action space
		for tup_path in observations:
			path = list(tup_path)
			cur_node = self.root
			for i in range(len(path)):
				a = path[i]
				
				if i == (len(path) - 1):
					comp_history = path[:]
					term = Node(None, (None, None), comp_history, None, 2)
					term.make_terminal(np.zeros(2))
					
					def gen_matching_nodes(children, comp_history):
						for x in children:
							if x.history == comp_history:
								yield x

					def gen_matching_terminal_nodes(terminal_nodes, comp_history):
						for x in terminal_nodes:
							if x.history[:(i+1)] == comp_history:
								yield x

					matching = list(gen_matching_nodes(cur_node.children, comp_history))
					matching_terminal_nodes = list(gen_matching_terminal_nodes(self.terminal_nodes, comp_history))
					
					# make sure there's no identical terminal nodes with matching history AND that 
					# this possible new terminal node's history isn't already covered by a decision node
					if len(matching) == 0 and len(matching_terminal_nodes) == 0:
						cur_node.add_children([term])
						self.terminal_nodes.append(term)

					if len(matching_terminal_nodes) > 0:
						for x in matching_terminal_nodes:
							payoffs[tuple(x.history)] = payoffs.get(tuple(x.history), []) + payoffs.get(tuple(comp_history), [])

					for j in range(i):
						h = path[:j+1]
						old_payoffs = payoffs.get(tuple(h), [])
						if old_payoffs != []:
							new_payoffs = payoffs.get(tuple(comp_history), []) + old_payoffs
							payoffs[tuple(comp_history)] = new_payoffs[:]
							del payoffs[tuple(h)]

					cur_infoset = self.get_infoset_given_node(cur_node)
					if a not in cur_infoset.action_space:
						cur_infoset.action_space += [a]
						for n in cur_infoset.node_list:
							n.action_space = cur_infoset.action_space[:]
					
					if a not in cur_node.action_space:
						cur_node.action_space += [a]
				
				elif i % 3 == 0 and len(a) == 1:
					def get_matching_children(children, a):
						for x in children:
							if x.history[-1] == a:
								yield x

					children_matching = list(get_matching_children(cur_node.children, a))
					if len(children_matching) == 0:
						# New path is being formulated
						cur_node.action_space += [a]
						if cur_node in self.chance_map.keys():
							cur_dist = self.chance_map[cur_node].copy()
							cur_dist[a] = cur_dist.get(a, 0.0) + observations.get(tup_path)
							self.chance_map[cur_node] = cur_dist
						else:
							self.chance_map[cur_node] = {a : observations.get(tup_path)}
						
						next_player_id = 1
						next_infoset_id = (1, tuple(path)[:i+1])
						next_node = Node(next_player_id, next_infoset_id, cur_node.history + [a], [], 2)
						cur_node.add_children([next_node])

						def gen_matching_infosets(infosets, next_infoset_id):
							for x in infosets:
								if x.infoset_id == next_infoset_id:
									yield x

						matching_infosets = list(gen_matching_infosets(self.infosets[0], next_infoset_id))
						if matching_infosets == []:
							next_infoset = Infoset(next_infoset_id, [next_node], [], 2)
							self.infosets[next_player_id - 1].append(next_infoset)
						else:
							for matching_infoset in matching_infosets:
								next_node.action_space = matching_infoset.action_space[:]
								matching_infoset.node_list += [next_node]

					else:
						path_to_match = path[:i+1]
						if cur_node in self.chance_map.keys():
							cur_dist = self.chance_map[cur_node].copy()
							cur_dist[a] = cur_dist.get(a, 0.0) + observations.get(tup_path)
							self.chance_map[cur_node] = cur_dist
						else:
							self.chance_map[cur_node] = {a : observations.get(tup_path)}

						def gen_next_nodes(children, a, path_to_match):
							for x in children:
								if a == x.history[-1] and list(x.infoset_id[1]) == path_to_match:
									yield x

						next_nodes = list(gen_next_nodes(cur_node.children, a, path_to_match))
						next_node = next_nodes[0]

					cur_node = next_node
							
				else:

					def get_matching_children(path, i, cur_node):
						for x in cur_node.children:
							if x.history == path[:(i+1)]:
								yield x

					def get_nonterminal_matching_children(children_matching):
						for x in children_matching:
							if x not in self.terminal_nodes:
								yield x

					children_matching = list(get_matching_children(path, i, cur_node))
					non_terminal_children_matching = list(get_nonterminal_matching_children(children_matching))
					cur_player_id = cur_node.infoset_id[0]
					
					if len(children_matching) == 0:
						# New path is being formulated
						assert cur_player_id != 0
						cur_infoset = self.get_infoset_given_node(cur_node)
						
						if a not in cur_infoset.action_space:
							cur_infoset.action_space.append(a)
							for n in cur_infoset.node_list:
								n.action_space = cur_infoset.action_space[:]

						if a not in cur_node.action_space:
							cur_node.action_space.append(a)

						next_infoset_id = self.get_next_infoset_id(path, i)
						
						def get_infoset_given_id(next_infoset_id):
							j = next_infoset_id[0]
							for x in self.infosets[j - 1]:
								if x.infoset_id == next_infoset_id:
									yield x

						if next_infoset_id == 0:
							next_infoset_id = (0, len(self.chance_map))
							next_node = Node(0, next_infoset_id, cur_node.history + [a], [], 2)
						else:
							next_player_id = next_infoset_id[0]
							next_node = Node(next_player_id, next_infoset_id, cur_node.history + [a], [], 2)
							matching_infosets = list(get_infoset_given_id(next_infoset_id))

							if matching_infosets == []:
								next_infoset = Infoset(next_infoset_id, [next_node], [], 2)
								self.infosets[next_player_id - 1].append(next_infoset)

							else:
								for matching_infoset in matching_infosets:
									next_node.action_space = matching_infoset.action_space[:]
									matching_infoset.node_list += [next_node]
								assert len(matching_infosets) == 1

						cur_node.add_children([next_node])

					elif len(non_terminal_children_matching) == 0:
						def gen_matching_children(children, a):
							for x in children:
								if x.history[-1] == a:
									yield x

						next_nodes = list(gen_matching_children(cur_node.children, a))
						assert len(next_nodes) == 1
						next_node = next_nodes[0]
						assert next_node in children_matching
						current_utility = next_node.utility

						next_infoset_id = self.get_next_infoset_id(path, i)
						
						if next_infoset_id == 0:
							next_infoset_id = (0, len(self.chance_map))

						next_player_id = next_infoset_id[0]
						next_node.is_terminal = False
						next_node.player_id = next_player_id
						next_node.infoset_id = next_infoset_id
						next_node.action_space = []
						self.terminal_nodes.remove(next_node)

						if next_player_id != 0:
							def get_infoset_given_id(next_infoset_id):
								j = next_infoset_id[0]
								for x in self.infosets[j - 1]:
									if x.infoset_id == next_infoset_id:
										yield x

							matching_infosets = list(get_infoset_given_id(next_infoset_id))
							if matching_infosets == []:
								next_infoset = Infoset(next_infoset_id, [next_node], [], 2)
								self.infosets[next_player_id - 1].append(next_infoset)
							else:
								assert len(matching_infosets) == 1
								for matching_infoset in matching_infosets:
									next_node.action_space = matching_infoset.action_space[:]
									matching_infoset.node_list += [next_node]

						if np.any(current_utility):
							payoffs[tuple(next_node.history)] = payoffs.get(tuple(next_node.history), []) + [current_utility]

					else:
						def gen_matching_children(children, a):
							for x in children:
								if x.history[-1] == a:
									yield x
						
						next_nodes = list(gen_matching_children(cur_node.children, a))
						assert len(next_nodes) == 1
						next_node = next_nodes[0]

						if cur_player_id == 0:
							cur_dist = self.chance_map[cur_node].copy()
							cur_dist[a] = cur_dist.get(a, 0.0) + 1
							self.chance_map[cur_node] = cur_dist

					cur_node = next_node

		for t in self.terminal_nodes:
			util = np.zeros(2)
			payoffs_t = [payoffs[x] for x in payoffs.keys() if x == tuple(t.history)]
			if payoffs_t != []:
				payoffs_t = [payoffs[x] for x in payoffs.keys() if x == tuple(t.history)][0]
				t.utility = np.sum(payoffs_t, axis=0) / len(payoffs_t)

	def subgame_cfr(self, T, game_root=None, partial_solution={}):
		'''
		General implementation of counterfactual regret minimization (CFR).
		This method is to be called by the empirical game ExtensiveForm object
		Returns a new metastrategy that should ultimately be an approx. NE

		Later: could modify to include beliefs?
		'''
		if game_root is None:
			game_root = self.root

		def gen_subgame_nodes(game_root):
			yield game_root
			children = game_root.children[:]
			while children != []:
				next_children = []
				for c in children:
					yield c
					if not c.is_terminal:
						next_children += c.children[:]
				children = next_children[:]

		def gen_subgame_infoset_ids(game_root, player_list):
			for x in gen_subgame_nodes(game_root):
				if x.player_id != 0 and not x.is_terminal and x.player_id in player_list:
					yield x.infoset_id

		def gen_nontrivial_subgame_infosets(game_root, partial_solution, player_list):
			for x in gen_subgame_infoset_ids(game_root, player_list):
				if x not in partial_solution:
					yield x

		def gen_subgame_infosets(game_root, j):
			for i in self.infosets[j]:

				if i.infoset_id in gen_subgame_infoset_ids(game_root, [j + 1]):
					yield i

		if game_root.player_id == 0:
			# speedup
			nontrivial_subgame_infosets = list(gen_nontrivial_subgame_infosets(game_root, partial_solution, [1, 2]))

			if nontrivial_subgame_infosets == []:
				return partial_solution

		for j in range(2):
			for infoset in gen_subgame_infosets(game_root, j):
				if infoset.infoset_id in partial_solution:
					num_actions = len(infoset.action_space)
					strat = partial_solution[infoset.infoset_id]
					x = []
					for i in range(num_actions):
						a = infoset.action_space[i]
						x.append(strat[a])
					infoset.set_strategy(np.array(x))
				else:
					num_actions = len(infoset.action_space)
					infoset.set_strategy(np.repeat(1.0 / num_actions, num_actions))
					infoset.regret_sum = np.zeros(num_actions)
					infoset.strategy_sum = np.zeros(num_actions)
					infoset.reach_prob_sum = 0
					infoset.reach_prob = 0
					infoset.action_utils = np.zeros((num_actions, 2))


		expected_val_cur_strategy = np.zeros(2)
		subgame_infosets_0 = list(gen_subgame_infosets(game_root, 0))
		subgame_infosets_1 = list(gen_subgame_infosets(game_root, 1))
		subgame_infosets = [subgame_infosets_0, subgame_infosets_1]
		for t in range(T):
			# added code to handle a partial solution
			expected_val_cur_strategy += self.recursive_cfr_helper(game_root, 1.0, 1.0, 1.0, partial_solution)
			for j in range(2):
				for infoset in subgame_infosets[j]:
					if infoset.infoset_id not in partial_solution:
						infoset.update_strategy()


		nash_strat = {}
		for j in range(2):
			for infoset in subgame_infosets[j]:
				if infoset.infoset_id not in partial_solution:
					nash_I = infoset.compute_average_strategy()
					num_actions = len(infoset.action_space)
					dist = {}
					for i in range(num_actions):
						a = infoset.action_space[i]
						dist[a] = nash_I[i].copy()

					nash_strat[infoset.infoset_id] = dist.copy()
				else:
					nash_strat[infoset.infoset_id] = partial_solution[infoset.infoset_id].copy()

		return nash_strat

	def get_prob_dist_given_chance_map(self, node):
		'''
		'''
		card_weights = self.chance_map.get(node).copy()
		prob_dist = {}
		denom = sum(card_weights.values())
		for e in card_weights.keys():
			prob_dist[e] = card_weights.get(e) / denom

		return prob_dist


	def cfr(self, T):
		'''
		General implementation of counterfactual regret minimization (CFR).
		This method is to be called by the empirical game ExtensiveForm object
		Returns a new metastrategy that should ultimately be an approx. NE

		Later: could modify to include beliefs?
		'''
		print("called regular CFR")

		# First, initialize key variables
		for j in range(2):
			for infoset in self.infosets[j]:
				num_actions = len(infoset.action_space)
				infoset.set_strategy(np.repeat(1.0 / num_actions, num_actions))
				infoset.regret_sum = np.zeros(num_actions)
				infoset.strategy_sum = np.zeros(num_actions)
				infoset.reach_prob_sum = 0
				infoset.reach_prob = 0
				infoset.action_utils = np.zeros((num_actions, 2))

		expected_val_cur_strategy = np.zeros(2)
		for t in range(T):
			if t % 100 == 0:
				print("t ", t)

			expected_val_cur_strategy += self.recursive_cfr_helper(self.root, 1.0, 1.0, 1.0)
			for j in range(2):
				for infoset in self.infosets[j]:
					infoset.update_strategy()

		# computing average strat --> Nash equil
		nash_strat = {}
		for j in range(2):
			for infoset in self.infosets[j]:
				nash_I = infoset.compute_average_strategy()
				num_actions = len(infoset.action_space)
				dist = {}
				for i in range(num_actions):
					a = infoset.action_space[i]
					dist[a] = nash_I[i]

				nash_strat[infoset.infoset_id] = dist

		return nash_strat

	def recursive_cfr_helper(self, current_node, player1_prob, player2_prob, chance_prob, partial_solution={}):
		'''
		@arg (Node) current_node: node within a current information set we are currently visiting as we
			play the game
		@arg (Infoset) current_infoset: the current information set we're visiting

		@arg (float) player1_prob: the reach probability contributed by player 1
		@arg (float) player2_prob: the reach probability contributed by player 2
		@arg (float) chance_prob: the reach probability contributed by Nature

		Recursive helper function that updates action utilities, computes the counterfactual utilities
			of the current strategy, and updates cumulative regret at that information set in turn
		'''
		if current_node.player_id == 0:
			expected_pay = np.zeros(2)
			prob_dist = self.get_prob_dist_given_chance_map(current_node)
			
			for outcome in prob_dist.keys():
				next_node = current_node.get_child_given_action(outcome)
				next_player_id = next_node.player_id
				next_infoset = self.get_infoset_given_node(next_node)
				next_pay = self.recursive_cfr_helper(next_node, player1_prob, player2_prob, chance_prob * prob_dist.get(outcome), partial_solution)
				expected_pay += next_pay * prob_dist.get(outcome) / sum(prob_dist.values())

			return expected_pay

		elif current_node.is_terminal:
			return current_node.utility

		elif current_node.infoset_id in partial_solution:
			return current_node.compute_pay(partial_solution, self.chance_map, 1.0)

		# Now we want to compute the counterfactual utility
		current_infoset = self.get_infoset_given_node(current_node)
		num_avail_actions = len(current_infoset.action_space)
		current_strategy = current_infoset.strategy

		# now try this to fix reach_prob being > 1.0 for player 2 at times
		if current_node.player_id == 1:
			current_infoset.reach_prob += player1_prob

		else:
			current_infoset.reach_prob += player2_prob

		infoset_action_utils = np.zeros((num_avail_actions, 2))
		for i in range(num_avail_actions):
			a = current_infoset.action_space[i]
			next_node = current_node.get_child_given_action(a)

			if next_node is not None:
				if current_node.player_id == 1:
					# updated player 1 
					infoset_action_utils[i] = self.recursive_cfr_helper(next_node, player1_prob * current_strategy[i], player2_prob, chance_prob, partial_solution)
				else:
					# updated player 2
					infoset_action_utils[i] = self.recursive_cfr_helper(next_node, player1_prob, player2_prob * current_strategy[i], chance_prob, partial_solution)

			else:
				if current_node.player_id == 1:
					infoset_action_utils[i] = np.array([0.0, 0.0])
				else:
					infoset_action_utils[i] = np.array([0.0, 0.0])

		# Now compute the total utility of the information set
		infoset_cfu = np.matmul(current_strategy, infoset_action_utils)

		# Compute the regrets of not playing each action at the infoset
		regrets = infoset_action_utils - infoset_cfu

		if current_node.player_id == 1:
			current_infoset.regret_sum += regrets[:, 0] * player2_prob * chance_prob
		else:
			current_infoset.regret_sum += regrets[:, 1] * player1_prob * chance_prob

		return infoset_cfu

	def compute_SPE(self, T, tree_of_roots, anynode_to_node_map):
		'''
		Algorithm for computing the subgame perfect equilibria (SPE) of the game.
		Note that since we will be pruning this game as we go, this method is called
		on a copy of the original game
		'''
		max_height = tree_of_roots.height + 1
		subgame_groups = self.get_subgame_groups(tree_of_roots, max_height, anynode_to_node_map, 1)
		SPE = self.get_initial_SPE(subgame_groups[1], T)

		for k in range(1, max_height):
			if SPE == {}:
				return None

			solution_k = SPE.copy()
			next_SPE = {}

			for g_theta_next in subgame_groups[k + 1]:
				solution_k_g_theta = self.restrict_solution_to_subgame(g_theta_next, solution_k)
				if g_theta_next.player_id != 0:
					del solution_k_g_theta[g_theta_next.infoset_id]
				
				solution_k_plus_1 = self.subgame_cfr(T, game_root=g_theta_next, partial_solution=solution_k_g_theta)
				next_SPE.update(solution_k_plus_1)

			del subgame_groups[k]
			del subgame_groups[k + 1]

			SPE.update(next_SPE)
			subgame_groups = self.get_subgame_groups(tree_of_roots, max_height, anynode_to_node_map, k + 1)

		for x in self.infosets[0] + self.infosets[1]:
			assert x.infoset_id in SPE.keys()
			assert SPE.get(x.infoset_id) is not None

		del subgame_groups

		return SPE

	def get_subgame_roots(self):
		'''
		Helper method that returns a subtree of the root nodes of all the game's subgames.
		Note: the root MUST consist of a single node, not an infoset with 2+ nodes
		'''
		# subtree will be identified by the root node, as its own AnyNode object
		subtree_of_roots = AnyNode(id=str(self.root.history))

		def get_root_children():
			for x in self.root.children:
				yield x

		# iterate over all decision nodes, starting with those closest to the root
		checked = [self.root]
		checking = {subtree_of_roots: get_root_children()}
		anynode_to_node_map = {}
		anynode_to_node_map[subtree_of_roots.root] = self.root

		def get_subgame_root_infoset(h):
			j = h.player_id
			for i in self.infosets[j - 1]:
				if i.infoset_id == h.infoset_id:
					yield i

		def get_subtree_nodes(h):
			yield h
			children = h.children[:]
			while children != []:
				next_children = []
				for c in children[:]:
					yield c
					if not c.is_terminal:
						next_children += c.children[:]
				children = next_children[:]

		def get_subtree_infosets(subtree_nodes):
			for x in subtree_nodes:
				if not x.is_terminal and x.player_id != 0:
					infoset = self.get_infoset_given_node(x)
					yield infoset

		while checking != {}:
			to_be_checked = {}
			for parent_subgame in checking:
				for h in checking[parent_subgame]:
					found_subgame_root = True
					subtree_nodes_gen = get_subtree_nodes(h)

					if h.player_id != 0:
						h_infoset = list(get_subgame_root_infoset(h))[0]
						is_singleton_infoset = len(h_infoset.node_list) == 1
						
						if not is_singleton_infoset:
							found_subgame_root = False
					
					for infoset in get_subtree_infosets(subtree_nodes_gen):
						for n in infoset.node_list:
							if n not in get_subtree_nodes(h):
								found_subgame_root = False

					def get_new_decision_nodes(node):
						for x in node.children:
							if not x.is_terminal:
								yield x

					if found_subgame_root:
						# we found a subgame root
						new_subgame = AnyNode(id=str(h.history), parent=parent_subgame)
						anynode_to_node_map[new_subgame] = h
						to_be_checked[new_subgame] = get_new_decision_nodes(h)
					else:
						to_be_checked[parent_subgame] = it.chain(to_be_checked.get(parent_subgame, []), get_new_decision_nodes(h))
							
					checked.append(h)

			checking = to_be_checked

		return subtree_of_roots, anynode_to_node_map

	def get_subgame_groups(self, subtree, max_height, anynode_to_node_map, k):
		'''
		@arg (AnyNode) subtree: subtree of subgame roots, given as an AnyNode root
		@arg (int) max_height: height of the full game's root in the subtree of roots
		@arg (dict: AnyNode -> Node) anynode_to_node_map: map from the AnyNode objects
			in the subtree of subgame roots to the corresponding roots (Node objects) in
			the full game tree

		Helper method that groups the game's subgame roots by their heights in the subtree \Psi
		'''
		all_theta = {}
		def gen_subgame_roots_at_k(subtree, k_):
			for h in (subtree.root,) + subtree.descendants:
				if h.height + 1 == k_:
					subgame_root = anynode_to_node_map[h]
					yield subgame_root
		
		all_theta[k] = gen_subgame_roots_at_k(subtree, k)
		all_theta[k + 1] = gen_subgame_roots_at_k(subtree, k + 1)

		return all_theta

	def get_initial_SPE(self, thetas_1, T):
		'''
		@arg (list of ExtensiveForm's) thetas_1: list of subgames in the true game whose roots
			are at height 1 in the tree of subgame roots (i.e. closest to the tree leaves)

		Helper method that finds the initial partial SPE for the subgames closest to the
		terminal nodes in the game; returns each subgame's solution as a single, collective solution
		'''
		SPE = {}
		for g_theta in thetas_1:
			soln = self.find_nash(g_theta, T)
			SPE.update(soln)

		return SPE

	def find_nash(self, g_theta, T):
		'''
		@arg (Node) g_theta: root of a subgame in the empirical game tree

		Helper method that computes the NE of the given subgame g_theta by converting it into
		sequence-form and then solving the resulting LCP using Lemke-Howson
		'''
		return self.subgame_cfr(T, g_theta)

	def restrict_solution_to_subgame(self, g_theta, solution_profile):
		'''
		@arg (Node) g_theta: root of a subgame in the empirical game tree
		@arg (map: tuple --> (map: str --> float)) solution_profile: strategy that maps
			each player infoset to a probability distribution over that infoset's corresponding
			action space

		Helper method that restricts the complete game's solution profile to just a specific subgame
		'''
		subgame_solution = {}
		if g_theta.player_id != 0:
			subgame_solution[g_theta.infoset_id] = solution_profile.get(g_theta.infoset_id)
		children = g_theta.children[:]
		while children != []:
			next_children = []
			for child in children:
				if not child.is_terminal:
					if child.player_id != 0 and child.infoset_id in solution_profile and child.infoset_id not in subgame_solution:
						subgame_solution[child.infoset_id] = solution_profile.get(child.infoset_id)
					
					next_children += child.children

			children = next_children[:]

		del children

		return subgame_solution

	def compute_pay_dp(self, subgame_solution, g_theta, pay_map, input_reach_prob):
		'''
		@arg (map: tuple --> (map: str --> float)) subgame_solution: strategy that maps
			each player infoset to a probability distribution over that infoset's corresponding
			action space; restricted to a particular subgame only
		@arg (Node) g_theta: root of a subgame in the empirical game tree
		@arg (map: Node --> np.array) pay_map: map from subgame roots to the pay from playing
			the corresponding portions of a given solution profile in that subgame
		@arg (float) input_reach_prob: probability of reaching this current subgame root g_theta

		Computes the payoff of playing the current subgame solution at this particular subgame; utilizes
		a dynamic programming approach to store/read the payoffs from smaller subgames contained within
		the current one, computed the same
		'''
		if g_theta in pay_map:
			return pay_map[g_theta] * input_reach_prob

		elif g_theta.is_terminal:
			return g_theta.utility * input_reach_prob

		elif g_theta.player_id == 0:
			pay = np.zeros(self.num_players)
			prob_dist = self.get_prob_dist_given_chance_map(g_theta)
			
			for outcome in prob_dist.keys():
				next_node = g_theta.get_child_given_action(outcome)
				next_reach_prob = input_reach_prob * prob_dist.get(outcome) / sum(prob_dist.values())
				pay = pay + self.compute_pay_dp(subgame_solution, next_node, pay_map, next_reach_prob)

		else:
			pay = np.zeros(self.num_players)			
			infoset_strat = subgame_solution.get(g_theta.infoset_id)
			
			if infoset_strat is not None:
				for a in infoset_strat.keys():
					next_node = g_theta.get_child_given_action(a)
					
					if next_node is not None:
						next_reach_prob = input_reach_prob * infoset_strat.get(a, 0.0)
						if next_reach_prob > 0.0:
							pay = pay + self.compute_pay_dp(subgame_solution, next_node, pay_map, next_reach_prob)

		return pay


	def compute_subgame_regret(self, solution_profile, g_theta, pay_map, regret_map):
		'''
		@arg (map: tuple --> (map: str --> float)) solution_profile: strategy that maps
			each player infoset to a probability distribution over that infoset's corresponding
			action space
		@arg (Node) g_theta: root of a subgame in the empirical game tree
		@arg (map: Node --> np.array) pay_map: map from subgame roots to the pay from playing
			the corresponding portions of a given solution profile in that subgame
		@arg (map: Node --> np.array) regret_map: map from subgame roots to the worst-case regret
			in those subgames in response to the other players' current strategy

		Algorithm for computing the regret within a subgame (self)
		4/20: Added dynamic programming modifications so that the generators operate over fewer actions
		and infosets within a given subgame
		'''
		def action_space_generator(j, subgame_solution, regret_map):
			sigma_pay_j = np.zeros(2)
			for infoset in self.infosets[j]:
				if infoset.infoset_id in subgame_solution and infoset.infoset_id not in regret_map:
					yield infoset.action_space


		def infoset_generator(j, subgame_solution, regret_map):
			r = 0
			for infoset in self.infosets[j]:
				if infoset.infoset_id in subgame_solution and infoset.infoset_id not in regret_map:
					yield infoset, r
					r += 1

		subgame_solution = self.restrict_solution_to_subgame(g_theta, solution_profile)
		regret = [-1000.0, -1000.0]
		solution_candidate_pay = self.compute_pay_dp(subgame_solution, g_theta, pay_map, 1.0)
		pay_map[g_theta] = solution_candidate_pay

		for j in range(self.num_players):
			regret_j = 0
			sigma_regret_j = {}
			solution_candidate_pay_j = solution_candidate_pay[j]
			
			# How to compute regret but restrict my considerations only to this subgame and not action combos
			# for a subgame further down for which I have already found the regret?
			for s_j in it.product(*action_space_generator(j, subgame_solution, regret_map)):

				if len(s_j) > 0:
					sigma_j = {}
					for infoset_index_pair in infoset_generator(j, subgame_solution, regret_map):
						i = infoset_index_pair[0]
						infoset_index = infoset_index_pair[1]
						sigma_j[i.infoset_id] = {s_j[infoset_index]: 1.0}
						sigma_regret_j = sigma_j.copy()

					# add other players' strategies from solution profile (sigma_{-j})
					for infoset_id in subgame_solution:
						if infoset_id not in sigma_j.keys():
							if infoset_id in regret_map and infoset_id[0] == (j + 1):
								sigma_j[infoset_id] = regret_map[infoset_id].copy()
							else:
								sigma_j[infoset_id] = subgame_solution[infoset_id].copy()

					sigma_pay_j = g_theta.compute_pay(sigma_j, self.chance_map, 1.0)[j]
					pay_diff_j = sigma_pay_j - solution_candidate_pay_j
					if pay_diff_j > regret_j:
						regret_j = pay_diff_j

						for infoset_id in sigma_regret_j:
							regret_map[infoset_id] = sigma_regret_j[infoset_id].copy()


			if regret_j == 0:
				for infoset_id in sigma_regret_j:
					regret_map[infoset_id] = subgame_solution[infoset_id].copy()

			regret[j] = max(regret_j, regret[j])

		return regret, pay_map, regret_map

	def compute_max_regret_across_subgames(self, solution_profile, tree_of_roots, anynode_to_node_map, max_height):
		'''
		@arg (map: tuple --> (map: str --> float)) solution_profile: strategy that maps
			each player infoset to a probability distribution over that infoset's corresponding
			action space
		@arg (map: int --> list of ExtensiveForm objects) subgame_groups: map of each possible
			subgame level (1 <= k <= \ell) in the game tree to the list of subgames at that level
		@arg (int) max_height: height of game root in tree of subgame roots; a.k.a. maximum subgame
			height \ell

		Algorithm for computing the worst-case subgame regret for a given solution across all subgames
		'''
		max_regret = [-1000.0, -1000.0]
		regret_k = {}
		subgame_groups = self.get_subgame_groups(tree_of_roots, max_height, anynode_to_node_map, 1)

		# map from subgame root to solution candidate pay
		pay_map = {}

		# map from subgame root to sigma_j pay that maximizes regret
		regret_map = {}

		if max_height == 1:
			regret_players, pay_map, regret_map = self.compute_subgame_regret(solution_profile, self.root, pay_map, regret_map)
			regret_k[(1, self.root.infoset_id)] = max(regret_players)

			return max(regret_k.values())

		for k in range(1, max_height):

			# 4/20: This was sped up with dynamic programming, the computing-regret and pay process
			for g_theta in subgame_groups[k]:
				if g_theta.player_id != 0:
					regret_players, pay_map, regret_map = self.compute_subgame_regret(solution_profile, g_theta, pay_map, regret_map)
					if max(regret_players) > 0:
						regret_k[(k, g_theta.infoset_id)] = max(regret_players)

			if len(regret_k) > 0:
				max_regret = max(regret_k.values())
			else:
				max_regret = 0.0

			del subgame_groups[k]
			del subgame_groups[k + 1]

			subgame_groups = self.get_subgame_groups(tree_of_roots, max_height, anynode_to_node_map, k + 1)

		del pay_map
		del regret_k
		gc.collect()

		return max_regret
