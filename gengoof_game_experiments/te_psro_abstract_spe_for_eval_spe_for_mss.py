import random
import string
import numpy as np
import math
import itertools as it
import copy
import gc
import shutil
import sys
import json
from Node import *
from Infoset import *
from ExtensiveForm import *
from DQN import *
from compute_memory import *
from best_response import *
from heapq import nlargest

N = 2
SAMPLING_BUDGET = 200
STD_DEV = 0.0
REGRET_THRESH = 0.0
NUM_PSRO_ITER = 20
T = 500

HANDLERS = {
	ExtensiveForm: ExtensiveFormHandler,
	Infoset: InfosetHandler,
	Node: NodeHandler
}

file_ID_list = {
4: ['abstract_4_0_35OWM', 
'abstract_4_1_3HMHG', 
'abstract_4_2_SQQI1', 
'abstract_4_0_M303W', 
'abstract_4_1_JGRYB', 
'abstract_4_2_Q1YW4', 
'abstract_4_0_JREIF', 
'abstract_4_1_0AFWK', 
'abstract_4_2_UQK51', 
'abstract_4_0_LI11S', 
'abstract_4_1_36I6G', 
'abstract_4_2_8C9ZC', 
'abstract_4_0_YSMOJ', 
'abstract_4_1_9U5VB', 
'abstract_4_2_NDGEW', 
'abstract_4_0_XJDR9', 
'abstract_4_1_31Q0G', 
'abstract_4_2_Y7R6S', 
'abstract_4_0_Z4Y9R', 
'abstract_4_1_QY80T', 
'abstract_4_2_R2WGO', 
'abstract_4_0_PJCXC', 
'abstract_4_1_Z77F7', 
'abstract_4_2_5DEAD', 
'abstract_4_0_3QTPV', 
'abstract_4_1_9FHX5', 
'abstract_4_2_XRVX3', 
'abstract_4_0_TDJN0', 
'abstract_4_1_V2UHF', 
'abstract_4_2_MYP70'
]

T_list = [
500, 
1000, 
2000, 
5000]

emp_br_list = [
1,
2,
4,
8,
16]

num_rounds_list = [
4]

def retrieve_game(file_ID_index, num_rounds):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	file_name = "game_parameters_" + str(num_rounds) + "_rounds.npz"
	a_f = np.load(file_name, allow_pickle=True)
	lst = a_f.files
	for params in a_f['arr_0']:
		if file_ID_index == params[0]:
			return params

def retrieve_json_hps(num_rounds, player_num):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration
	@arg (int) player_num: index {1, 2} corresponding to one of the two players

	Helper method to retrieve each set of learned hyperparameter values from
	phases 1 and 2 for a given game and player
	'''
	with open('optimal_learned_hp_abstract.json') as f:
		data = f.read()

	js = json.loads(data)
	d_both = js[str(num_rounds)]
	d = d_both[str(player_num)]
	keys = d.keys()
	hp_set = None
	for elt in it.product(*d.values()):
		hp_set = dict(zip(keys, elt))

	return hp_set

def generate_new_BR_paths(empir_strat_space, BR):
	'''
	'''
	NUM_BR = 2**len(BR)

	# get every possible combination of BR's to be included in tree
	for j in range(1, NUM_BR):
		strat = {}
		bin_list = [int(x) for x in bin(j)[2:]]
		
		if len(bin_list) < len(BR):
			bin_list_copy = bin_list[:]
			bin_list = [0] * (len(BR) - len(bin_list_copy)) + bin_list_copy

		br_list = list(it.compress(BR, bin_list))
		for infoset_id in empir_strat_space:
			if infoset_id in br_list:
				strat[infoset_id] = [BR[infoset_id]]
			else:
				strat[infoset_id] = empir_strat_space[infoset_id][:]

		yield strat

	if NUM_BR == 1:
		strat = {}
		for infoset_id in empir_strat_space:
			strat[infoset_id] = empir_strat_space[infoset_id][:]

		yield strat

def get_total_nf_budget(SAMPLING_BUDGET, complete_psro_iter):
	'''
	'''
	num_cells_square = complete_psro_iter**2
	num_new_cells_square = (complete_psro_iter + 1)**2

	return (num_new_cells_square - num_cells_square) * SAMPLING_BUDGET


def simulate(game_param_map, old_strategy_space, BR, total_NF_sample_budget, noise, payoffs, POLICY_SPACE1, POLICY_SPACE2, 
	default_policy1, default_policy2):
	'''
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (map: Infoset --> (map: str --> float)) strategy_profile: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (int) num_iter: number of iterations
	@arg (float) noise: Gaussian noise added to the utility samples outputted at the end
		of a single simulation
	@arg (dict) payoffs: dictionary mapping histories to sampled utilities

	Black-box simulator that will be used for EGTA. Simulates a single run through the
	true game, num_iter times. Returns the observations and utilities returned for each run.

	Note: Simulates pure strategies ONLY
	'''
	observations = {}
	strategy_space = old_strategy_space.copy()
	num_tree_paths = max(2**len(BR) - 1, 1)
	num_iter = int(float(total_NF_sample_budget) / num_tree_paths)

	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	for strategy in generate_new_BR_paths(old_strategy_space, BR):
		for n in range(num_iter):
			action_history = []
			empir_history = []
			policy_history = []
			reached_end = False

			for r in range(num_rounds - 1):
				outcome, event_index = sample_stochastic_event_given_history(tuple(action_history), chance_events, card_weights)
				action_history += [outcome]
				if r in included_rounds:
					policy_history += [outcome]
					if not reached_end:
						empir_history += [outcome]

				empir_infoset_id1 = get_empirical_infoset_id_given_empir_history(policy_history, 1)
				p1_empir_action = None
				last_policy_str = None

				if empir_infoset_id1 not in strategy:
					if r in included_rounds:
						last_action = policy_history[-3]
						last_policy_str = get_last_policy_str(last_action, action_history, p1_actions, POLICY_SPACE1, game_params)
						p1_empir_action = get_action_given_policy(last_policy_str, p1_actions, POLICY_SPACE2, action_history, game_params)
					else:
						last_action = policy_history[-2]
						if last_action in p1_actions:
							p1_empir_action = get_last_policy_str(last_action, action_history, p1_actions, POLICY_SPACE1, game_params)
						else:
							p1_empir_action = last_action

					if last_policy_str is None:
						last_policy_str = default_policy1
						p1_empir_action = last_policy_str

					if not reached_end:
						reached_end = True
						empir_history += [p1_empir_action]

				else:
					p1_empir_action = random.choice(strategy.get(empir_infoset_id1))
					empir_history += [p1_empir_action]

				policy_history += [p1_empir_action]
				p1_action = None
				
				if r in included_rounds:
					p1_action = p1_empir_action
				else:
					p1_action = get_action_given_policy(p1_empir_action, p1_actions, POLICY_SPACE1, tuple(action_history), game_params)
				
				action_history += [p1_action]
				
				empir_infoset_id2 = get_empirical_infoset_id_given_empir_history(policy_history, 2)
				p2_empir_action = None
				
				if empir_infoset_id2 not in strategy:
					if r in included_rounds:
						last_action = policy_history[-3]
						last_policy_str = get_last_policy_str(last_action, action_history, p2_actions, POLICY_SPACE2, game_params)
						p2_empir_action = get_action_given_policy(last_policy_str, p2_actions, POLICY_SPACE2, action_history, game_params)
					else:
						last_action = policy_history[-2]
						#print("last_action2! ", last_action)
						if last_action in p2_actions:
							p2_empir_action = get_last_policy_str(last_action, action_history, p2_actions, POLICY_SPACE2, game_params)
						else:
							p2_empir_action = last_action

					if last_policy_str is None:
						last_policy_str = default_policy2
						p2_empir_action = last_policy_str

					if not reached_end:
						reached_end = True
						empir_history += [p2_empir_action]

				else:
					p2_empir_action = random.choice(strategy.get(empir_infoset_id2))
					empir_history += [p2_empir_action]

				policy_history += [p2_empir_action]
				p2_action = None
				if r in included_rounds:
					p2_action = p2_empir_action
				else:
					p2_action = get_action_given_policy(p2_empir_action, p2_actions, POLICY_SPACE1, tuple(action_history), game_params)					
				
				action_history += [p2_action]

			utility = get_utility(action_history, num_rounds, payoff_map)
			observations[tuple(empir_history)] = observations.get(tuple(empir_history), 0.0) + 1
			payoff_sample = np.random.normal(utility, np.array([noise] * 2))
			payoffs[tuple(empir_history)] = payoffs.get(tuple(empir_history), []) + [payoff_sample]

	return payoffs, observations

def compute_true_empirical_strategy_pay(meta_strategy, game_param_map, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	return recursive_true_empirical_strategy_pay_helper((), (), meta_strategy, 1.0, game_param_map, POLICY_SPACE1, POLICY_SPACE2)

def recursive_true_empirical_strategy_pay_helper(action_history, empir_history, strategy_profile, input_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	round_num = math.floor(len(action_history) / 3)
	num_p2_actions = len([x for x in action_history if x in p2_actions])
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	# End of game has been reached
	if num_p2_actions == (num_rounds - 1):
		util_vec = get_utility(list(action_history), num_rounds, payoff_map)
		return util_vec * input_reach_prob

	# chance node has been reached
	elif len(action_history) % 3 == 0:
		pay = np.zeros(N)
		chance_dist = get_chance_node_dist_given_history(action_history, chance_events, card_weights)
		for e in chance_dist.keys():
			prob = chance_dist.get(e)
			next_node = action_history + (e,)
			next_empir_history = empir_history
			if round_num in included_rounds:
				next_empir_history = empir_history + (e,)

			next_reach_prob = input_reach_prob * prob
			new_pay = recursive_true_empirical_strategy_pay_helper(next_node, next_empir_history, strategy_profile, next_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

	else:
		pay = np.zeros(N)
		PS = POLICY_SPACE1.copy()
		action_space = p1_actions[:]
		player_num = len(action_history) % 3
		if player_num == 2:
			PS = POLICY_SPACE2.copy()
			action_space = p2_actions[:]

		empir_infoset_id = get_empirical_infoset_id_given_empir_history(empir_history, player_num)
		infoset_strat = strategy_profile.get(empir_infoset_id)

		if infoset_strat is not None:
			for empir_action in infoset_strat.keys():
				prob = infoset_strat.get(empir_action, 0.0)
				if prob > 0.0:
					action = None
					if round_num in included_rounds:
						action = empir_action
					else:
						action = get_action_given_policy(empir_action, action_space, PS, action_history, game_params)	
					
					next_node = action_history + (action,)
					next_empir_history = empir_history + (empir_action,)
					next_reach_prob = input_reach_prob * prob
					new_pay = recursive_true_empirical_strategy_pay_helper(next_node, next_empir_history, strategy_profile, next_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay	

		else:
			empir_action = None
			last_policy_str = None
			
			if round_num in included_rounds:
				last_action = empir_infoset_id[1][player_num - 4] # -2 for player 2, -3 for player 1
				last_policy_str = get_last_policy_str(last_action, action_history, action_space, PS, game_params)

				empir_action = get_action_given_policy(last_policy_str, action_space, PS, action_history, game_params)
			else:
				last_action = empir_infoset_id[1][player_num - 3] # -1 for player 2, -2 for player 1
				if last_action in action_space:
					last_policy_str = get_last_policy_str(last_action, action_history, action_space, PS, game_params)
				else:
					last_policy_str = last_action
					
				empir_action = last_policy_str

			if last_policy_str is None:
				last_policy_str = "pi_0"
				empir_action = last_policy_str
				
			next_empir_history = empir_history + (empir_action,)
			action = None
			if round_num in included_rounds:
				action = empir_action
			else:
				action = get_action_given_policy(empir_action, action_space, PS, action_history, game_params)
			
			next_node = action_history + (action,)
			next_reach_prob = input_reach_prob
			new_pay = recursive_true_empirical_strategy_pay_helper(next_node, next_empir_history, strategy_profile, next_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

	return pay			

def compute_true_pay(empirical_strategy_profile, BR_network_weights, j, game_param_map, POLICY_SPACE1, POLICY_SPACE2):
	'''
	Computes the payoff of playing a given strategy profile in the true game "tree"
	'''
	return recursive_true_pay_helper((), (), empirical_strategy_profile, BR_network_weights, j, 1.0, game_param_map, POLICY_SPACE1, POLICY_SPACE2)

def recursive_true_pay_helper(action_history, empir_history, strategy_profile, BR_network_weights, br_player, input_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2):
	'''
	@arg (list) history: current node's history (i.e. how we identify them)
	@arg (map) strategy_profile: strategy profile
	@arg (float) input_reach_prob: probability of reaching the current node
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (list of int's) v1: player 1's valuation for each item in the pool
	@arg (list of int's) v2: player 2's valuation for each item in the pool

	Helper function that recursively travels the true game "tree" as we compute
	the payoff of a given strategy profile; meant to replace the same method
	in our Node class
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	round_num = math.floor(len(action_history) / 3)
	num_p2_actions = len([x for x in action_history if x in p2_actions])
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)
	
	# End of game has been reached
	if num_p2_actions == (num_rounds - 1):
		util_vec = get_utility(list(action_history), num_rounds, payoff_map)
		return util_vec * input_reach_prob

	# chance node has been reached
	elif len(action_history) % 3 == 0:
		pay = np.zeros(N)
		chance_dist = get_chance_node_dist_given_history(action_history, chance_events, card_weights)
		for e in chance_dist.keys():
			prob = chance_dist.get(e)
			next_node = action_history + (e,)
			next_empir_history = empir_history
			if round_num in included_rounds:
				next_empir_history = empir_history + (e,)
			next_reach_prob = input_reach_prob * prob
			new_pay = recursive_true_pay_helper(next_node, next_empir_history, strategy_profile, BR_network_weights, br_player, next_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay		

	else:
		pay = np.zeros(N)
		PS = POLICY_SPACE1.copy()
		action_space = p1_actions[:]

		player_num = len(action_history) % 3
		if player_num == 2:
			PS = POLICY_SPACE2.copy()
			action_space = p2_actions[:]

		if player_num != br_player:
			empir_infoset_id = get_empirical_infoset_id_given_empir_history(empir_history, player_num)
			infoset_strat = strategy_profile.get(empir_infoset_id)
			
			if infoset_strat is not None:
				for empir_action in infoset_strat.keys():
					next_empir_history = empir_history + (empir_action,)
					prob = infoset_strat.get(empir_action, 0.0)
					
					if prob > 0.0:
						action = None
						if round_num in included_rounds:
							action = empir_action
						else:
							action = get_action_given_policy(empir_action, action_space, PS, action_history, game_params)
						
						next_node = action_history + (action,)
						next_reach_prob = input_reach_prob * prob
						new_pay = recursive_true_pay_helper(next_node, next_empir_history, strategy_profile, BR_network_weights, br_player, next_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
						pay = pay + new_pay

			else:
				empir_action = None
				last_policy_str = None
				if round_num in included_rounds:
					last_action = empir_infoset_id[1][player_num - 4] # -2 for player 2, -3 for player 1
					last_policy_str = get_last_policy_str(last_action, action_history, action_space, PS, game_params)
					empir_action = get_action_given_policy(last_policy_str, action_space, PS, action_history, game_params)
				else:
					last_action = empir_infoset_id[1][player_num - 3] # -1 for player 2, -2 for player 1
					if last_action in action_space:
						empir_action = get_last_policy_str(last_action, action_history, action_space, PS, game_params)
					else:
						empir_action = last_action

				if last_policy_str is None:
					last_policy_str = "pi_0"
					empir_action = last_policy_str
				
				next_empir_history = empir_history + (empir_action,)
				action = None
				if round_num in included_rounds:
					action = empir_action
				else:
					action = get_action_given_policy(empir_action, action_space, PS, action_history, game_params)
				
				next_node = action_history + (action,)
				next_reach_prob = input_reach_prob
				new_pay = recursive_true_pay_helper(next_node, next_empir_history, strategy_profile, BR_network_weights, br_player, next_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay	
		else:
			state = convert_into_state(action_history, num_rounds, p1_actions, p2_actions, chance_events)
			best_action = get_best_action(state, BR_network_weights, action_space)
			next_empir_history = None
			
			if round_num in included_rounds:
				next_empir_history = empir_history + (best_action,)
			else:
				next_empir_history = empir_history + ("pi_" + str(len(PS)),)

			next_node = action_history + (best_action,)
			next_reach_prob = input_reach_prob
			new_pay = recursive_true_pay_helper(next_node, next_empir_history, strategy_profile, BR_network_weights, br_player, next_reach_prob, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

	return pay

def compute_regret(meta_strategy, eval_string, game_param_map, file_ID, hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
	'''
	@arg (map) ms: given strategy profile
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (str) file_ID: identification string for file containing outputs (error, regret, etc.)
	@arg (map) hp_set1: learned hyperparameters for player 1's DQN for best response
	@arg (map) hp_set2: learned hyperparameters for player 2's DQN for best response

	Computes regret for both players for a given strategy profile and returns the higher of the two regrets
	'''
	meta_strategy_pay = compute_true_empirical_strategy_pay(meta_strategy, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
	regrets = []
	BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(meta_strategy, eval_string, file_ID, game_param_map,
		hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, True)

	for j in range(2):
		action_pay = None
		if j == 0:
			action_pay = compute_true_pay(meta_strategy, BR1_weights, 1, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
		else:
			action_pay = compute_true_pay(meta_strategy, BR2_weights, 2, game_param_map, POLICY_SPACE1, POLICY_SPACE2)

		regrets.append(max(action_pay[j] - meta_strategy_pay[j], 0.0))

	return max(regrets)

def construct_initial_policy(game_param_map, hp_set1, hp_set2):
	'''
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	initial_policy = {}
	POLICY_SPACE1 = {}
	POLICY_SPACE2 = {}
	state_len = get_state_length(num_rounds)

	action_space1 = p1_actions[:]
	default_model1 = tf.keras.Sequential()
	default_model1.add(tf.keras.layers.Dense(hp_set1["model_width"], input_shape=(state_len,), activation="relu"))
	default_model1.add(tf.keras.layers.Dense(hp_set1["model_width"], activation="relu"))
	default_model1.add(tf.keras.layers.Dense(len(action_space1)))
	default_model1.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set1["learning_rate"]))
	W1_default = default_model1.get_weights()

	action_space2 = p2_actions[:]
	default_model2 = tf.keras.Sequential()
	default_model2.add(tf.keras.layers.Dense(hp_set2["model_width"], input_shape=(state_len,), activation="relu"))
	default_model2.add(tf.keras.layers.Dense(hp_set2["model_width"], activation="relu"))
	default_model2.add(tf.keras.layers.Dense(len(action_space2)))
	default_model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set2["learning_rate"]))
	W2_default = default_model2.get_weights()

	POLICY_SPACE1["pi_0"] = W1_default
	POLICY_SPACE2["pi_0"] = W2_default

	for card in chance_events:
		a1 = get_action_given_policy("pi_0", p1_actions, POLICY_SPACE1, (card,), game_params)
		initial_policy[(1, (card,))] = {a1: 1.0}
		a2 = get_action_given_policy("pi_0", p2_actions, POLICY_SPACE2, (card, a1,), game_params)
		initial_policy[(2, (card,))] = {a2: 1.0}

	return initial_policy, "pi_0", "pi_0", POLICY_SPACE1, POLICY_SPACE2

def compute_empirical_pay_given_infoset(br_meta_strat, infoset_id, game_param_map, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	return recursive_empirical_pay_infoset_helper(br_meta_strat, (), (), 1.0, 1.0, infoset_id, game_param_map, POLICY_SPACE1, POLICY_SPACE2)

def recursive_empirical_pay_infoset_helper(br_meta_strat, true_history, empir_history, input_reach_prob, infoset_reach_prob, infoset_id, game_param_map, 
	POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	infoset_freq = None
	round_num = math.floor(len(true_history) / 3)
	num_p2_actions = len([x for x in true_history if x in p2_actions])
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)
	
	if num_p2_actions == (num_rounds - 1):
		br_player = infoset_id[0]
		util_vec = get_utility(list(true_history), num_rounds, payoff_map)
		return util_vec[br_player - 1] * input_reach_prob, infoset_reach_prob
		

	elif len(true_history) % 3 == 0:
		pay = 0.0
		chance_dist = get_chance_node_dist_given_history(true_history, chance_events, card_weights)
		infoset_chance_events = [e for e in infoset_id[1] if e in chance_events]
		
		if round_num in included_rounds and round_num < len(infoset_chance_events):
			infoset_freq = 0.0
			# chance event is part of empirical infoset_id --> must be deterministic
			e = infoset_chance_events[round_num]
			prob = chance_dist.get(e)
			next_node = true_history + (e,)
			next_infoset_reach_prob = infoset_reach_prob * prob
			next_reach_prob = input_reach_prob
			next_empir_history = empir_history + (e,)
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, game_param_map, 
				POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			infoset_freq = new_infoset_freq

		else:
			infoset_freq = 0.0
			for e in chance_dist.keys():
				prob = chance_dist.get(e)
				next_node = true_history + (e,)
				next_empir_history = empir_history
				if round_num in included_rounds:
					next_empir_history = empir_history + (e,)
				next_infoset_reach_prob = infoset_reach_prob
				next_reach_prob = input_reach_prob * prob
				new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, game_param_map,
					POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				infoset_freq = new_infoset_freq

	else:
		player_num = len(true_history) % 3
		br_player = infoset_id[0]
		pay = 0.0
		infoset_freq = 0.0
		
		empir_infoset_id = get_empirical_infoset_id_given_empir_history(empir_history, player_num)
		input_empir_actions = [a for a in infoset_id[1]]
		cur_empir_actions = [a for a in empir_history]

		action_space = p1_actions[:]
		PS = POLICY_SPACE1
		if player_num == 2:
			action_space = p2_actions[:]
			PS = POLICY_SPACE2

		if len(cur_empir_actions) < len(input_empir_actions):
			# Choose BR Player's actions so they lead to the given infoset
			infoset_freq = 0.0
			action_index = len(cur_empir_actions)
			empir_action = input_empir_actions[action_index]
			action = None
			if round_num in included_rounds:
				action = empir_action
			else:
				action = get_action_given_policy(empir_action, action_space, PS, true_history, game_params)
			
			next_reach_prob = input_reach_prob
			next_node = true_history + (action,)
			next_empir_history = empir_history + (empir_action,)
			infoset_strat = br_meta_strat.get(empir_infoset_id)
			prob = infoset_strat.get(empir_action, 0.0)
			next_infoset_reach_prob = infoset_reach_prob * prob
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, game_param_map, 
				POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			infoset_freq = infoset_freq + new_infoset_freq

		else:
			infoset_strat = br_meta_strat.get(empir_infoset_id)

			if infoset_strat is not None:
				infoset_freq = 0.0
				for empir_action in infoset_strat.keys():
					prob = infoset_strat.get(empir_action, 0.0)
					action = None
					if round_num in included_rounds:
						action = empir_action
					else:
						action = get_action_given_policy(empir_action, action_space, PS, true_history, game_params)

					next_node = true_history + (action,)
					next_empir_history = empir_history + (empir_action,)
					next_reach_prob = input_reach_prob * prob
					next_infoset_reach_prob = infoset_reach_prob
					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, game_param_map, 
						POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					infoset_freq = new_infoset_freq

			else:
				infoset_freq = 0.0
				empir_action = None
				last_policy_str = None
				if round_num in included_rounds:
					last_action = empir_infoset_id[1][player_num - 4] # -2 for player 2, -3 for player 1
					last_policy_str = get_last_policy_str(last_action, true_history, action_space, PS, game_params)
					empir_action = get_action_given_policy(last_policy_str, action_space, PS, true_history, game_params)
				else:
					last_action = empir_infoset_id[1][player_num - 3] # -1 for player 2, -2 for player 1
					if last_action in action_space:
						last_policy_str = get_last_policy_str(last_action, true_history, action_space, PS, game_params)
						empir_action = last_policy_str
					else:
						last_policy_str = last_action
						empir_action = last_policy_str
				
				if last_policy_str is None:
					last_policy_str = "pi_0"
					empir_action = last_policy_str
				
				action = None
				if round_num in included_rounds:
					action = empir_action
				else:
					action = get_action_given_policy(empir_action, action_space, PS, true_history, game_params)
				
				next_node = true_history + (action,)
				next_empir_history = empir_history + (empir_action,)
				next_reach_prob = input_reach_prob
				next_infoset_reach_prob = infoset_reach_prob
				new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, game_param_map, 
					POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				infoset_freq = new_infoset_freq

	return pay, infoset_freq

def recursive_infoset_gain_helper(br_meta_strat, true_history, empir_history, input_reach_prob, infoset_id, br_player, BR_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	round_num = math.floor(len(true_history) / 3)
	num_p2_actions = len([x for x in true_history if x in p2_actions])
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	if num_p2_actions == (num_rounds - 1):
		br_player = infoset_id[0]
		util_vec = get_utility(list(true_history), num_rounds, payoff_map)
		return util_vec[br_player - 1] * input_reach_prob

	elif len(true_history) % 3 == 0:
		pay = 0.0
		chance_dist = get_chance_node_dist_given_history(true_history, chance_events, card_weights)
		infoset_chance_events = [e for e in infoset_id[1] if e in chance_events]
		
		if round_num in included_rounds and round_num < len(infoset_chance_events):
			e = infoset_chance_events[round_num]
			prob = chance_dist.get(e)
			next_node = true_history + (e,)
			next_reach_prob = input_reach_prob
			next_empir_history = empir_history + (e,)
			new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, game_param_map, 
				POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

		else:
			for e in chance_dist.keys():
				prob = chance_dist.get(e)
				next_empir_history = empir_history
				if round_num in included_rounds:
					next_empir_history = empir_history + (e,)

				next_node = true_history + (e,)
				next_reach_prob = input_reach_prob * prob
				
				new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, game_param_map, 
					POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay

	else:
		player_num = len(true_history) % 3
		br_player = infoset_id[0]
		pay = 0.0
		empir_infoset_id = get_empirical_infoset_id_given_empir_history(empir_history, player_num)
		input_empir_actions = [a for a in infoset_id[1]]
		cur_empir_actions = [a for a in empir_history]

		action_space = p1_actions[:]
		PS = POLICY_SPACE1
		if player_num == 2:
			action_space = p2_actions[:]
			PS = POLICY_SPACE2

		if len(cur_empir_actions) < len(input_empir_actions):
			action_index = len(cur_empir_actions)
			empir_action = input_empir_actions[action_index]
			action = None
			if round_num in included_rounds:
				action = empir_action
			else:
				action = get_action_given_policy(empir_action, action_space, PS, true_history, game_params)

			next_reach_prob = input_reach_prob
			next_node = true_history + (action,)
			next_empir_history = empir_history + (empir_action,)
			infoset_strat = br_meta_strat.get(empir_infoset_id)
			prob = infoset_strat.get(empir_action, 0.0)
			new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, game_param_map, 
				POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
		
		else:
			if player_num != br_player:
				infoset_strat = br_meta_strat.get(empir_infoset_id)
				
				if infoset_strat is not None:
					for empir_action in infoset_strat.keys():
						prob = infoset_strat.get(empir_action, 0.0)
						action = None
						
						if round_num in included_rounds:
							action = empir_action
						else:
							action = get_action_given_policy(empir_action, action_space, PS, true_history, game_params)
						
						next_node = true_history + (action,)
						next_empir_history = empir_history + (empir_action,)
						next_reach_prob = input_reach_prob * prob
						new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, game_param_map, 
							POLICY_SPACE1, POLICY_SPACE2)
						pay = pay + new_pay

				else:
					empir_action = None
					last_policy_str = None
					
					if round_num in included_rounds:
						last_action = empir_infoset_id[1][player_num - 4] # -2 for player 2, -3 for player 1
						last_policy_str = get_last_policy_str(last_action, true_history, action_space, PS, game_params)
						empir_action = get_action_given_policy(last_policy_str, action_space, PS, true_history, game_params)
					else:
						last_action = empir_infoset_id[1][player_num - 3] # -1 for player 2, -2 for player 1
						if last_action in action_space:
							last_policy_str = get_last_policy_str(last_action, true_history, action_space, PS, game_params)
							empir_action = last_policy_str

						else:
							last_policy_str = last_action
							empir_action = last_policy_str
					
					if last_policy_str is None:
						last_policy_str = "pi_0"
						empir_action = "pi_0"

					action = None
					if round_num in included_rounds:
						action = empir_action
					else:
						action = get_action_given_policy(empir_action, action_space, PS, true_history, game_params)	

					next_node = true_history + (action,)
					next_empir_history = empir_history + (empir_action,)
					next_reach_prob = input_reach_prob
					new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, game_param_map, 
						POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

			else:
				state = convert_into_state(true_history, num_rounds, p1_actions, p2_actions, chance_events)
				best_action = get_best_action(state, BR_weights, action_space)
				next_node = true_history + (best_action,)
				next_reach_prob = input_reach_prob
				next_empir_history = None
				policy_str = "pi_" + str(len(PS) - 1)
				
				if round_num in included_rounds:
					next_empir_history = empir_history + (best_action,)
				else:
					policy_str = "pi_" + str(len(PS) - 1)
					next_empir_history = empir_history + (policy_str,)
				
				new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, game_param_map, 
					POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay

	return pay

def compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, br_player, BR_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	empirical_pay, infoset_freq = compute_empirical_pay_given_infoset(br_meta_strat, infoset_id, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
	br_pay = recursive_infoset_gain_helper(br_meta_strat, (), (), 1.0, infoset_id, br_player, BR_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
	gain = br_pay - empirical_pay
	
	return gain, infoset_freq

def te_egta(game_param_map, T, trial_index, br_mss, eval_strat):
	'''
	@arg (map) initial_sigma: Initial metastrategy based on empirical strategy
		space
	@arg (map) game_param_map: map of game parameters for given file ID (player valuation distribution, item pool,
		outside offer distributions)
	@arg (int) T: Number of iterations for a single run of CFR (whether solving a game for NE or a subgame as part
		of solving a game for SPE)
	@arg (str) br_mss: identification for which solution type we will use as the MSS to find best responses
		for -- at present, either "ne" or "spe"
	@arg (str) eval_strat: identification for which solution type we will use as the strategy against which to
		compute true game regret and worst-case subgame regret in the empirical game -- at present either "ne"
		or "spe"

	Runs a single play of TE-EGTA on large abstract game, expanding strategy space, simulating
	each new strategy and constructing the empirical game model, which is then solved for an
	approximate NE using counter-factual regret minimization (CFR)
	'''
	file_ID = game_param_map["file_ID"]
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]

	file_ID = file_ID + "_" + str(included_rounds[-1])

	#extract learned hyperparameters for DQN
	hp_set1 = retrieve_json_hps(num_rounds, 1)
	hp_set2 = retrieve_json_hps(num_rounds, 2)

	empir_root = Node(0, (0, 1), [], [], N)
	X = {}
	initial_sigma, default_policy1, default_policy2, POLICY_SPACE1, POLICY_SPACE2 = construct_initial_policy(game_param_map, hp_set1, hp_set2)
	
	# Initialize the empirical strategy space based on initial_sigma
	empir_strat_space = {}
	for i in initial_sigma.keys():
		empir_strat_space[i] = [list(initial_sigma[i].keys())[0]]

	trial_index = "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
	prefix = 'NUM_EMPIR_BR' + str(NUM_EMPIR_BR) + '_' + str(trial_index) + '_' + file_ID + '_' + br_mss + '_mss_' + eval_strat + '_eval'
	file_name = prefix + '_empirical_game.mmdpickle'
	
	with open(file_name, 'w') as fp:
		pass

	mgame = mmp.mmapdict(file_name)
	empirical_game = ExtensiveForm([[], []], empir_root, [], {}, num_rounds)
	mgame['game'] = empirical_game
	
	regret_over_time = []
	max_subgame_regret_over_time = []
	br1_payoffs_over_time = []
	br2_payoffs_over_time = []
	ne_over_time = []
	spe_over_time = []

	payoffs = {}
	observations = []

	br_meta_strat = initial_sigma.copy()
	regret_meta_strat = initial_sigma.copy()
	empirical_game_size_over_time = []

	need_NE = (eval_strat == "NE") or (br_mss == "NE")
	need_SPE = (eval_strat == "SPE") or (br_mss == "SPE")

	eval_string = eval_strat + "_eval"
	mss_string = br_mss + "_mss"

	regret = compute_regret(regret_meta_strat, eval_string, game_param_map, prefix, hp_set1, hp_set2, POLICY_SPACE1.copy(), POLICY_SPACE2.copy(), default_policy1, default_policy2)
	regret_over_time.append(regret)
	print("regret_over_time so far ", regret_over_time)

	BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(br_meta_strat, mss_string, prefix, game_param_map, hp_set1, hp_set2, 
		POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, False)

	player1_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 1]
	num_br_samples = min(NUM_EMPIR_BR, len(player1_empir_infostates))
	policy_str = "pi_" + str(len(POLICY_SPACE1))
	POLICY_SPACE1[policy_str] = BR1_weights

	infoset_gains1 = []
	for infoset_id in player1_empir_infostates:
		infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 1, BR1_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
	
		if infoset_freq * infoset_gain > 0:
			infoset_gains1.append(infoset_gain * infoset_freq)
		else:
			infoset_gains1.append(-20000.0)

	x = np.arange(len(player1_empir_infostates))
	infoset_inds_1 = None
	try:
		infoset_inds_1 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains1))
	except:
		num_nonzero = len([y for y in softmax(infoset_gains1) if y > 0.0])
		infoset_inds_1 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains1))
	
	player1_empir_M = [player1_empir_infostates[i] for i in infoset_inds_1]
	BR1 = convert_into_best_response_policy(player1_empir_M, policy_str, BR1_weights, game_param_map)

	infoset_gains2 = []
	player2_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 2]
	num_br_samples = min(NUM_EMPIR_BR, len(player2_empir_infostates))
	policy_str = "pi_" + str(len(POLICY_SPACE2))
	POLICY_SPACE2[policy_str] = BR2_weights
	for infoset_id in player2_empir_infostates:
		infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 2, BR2_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
	
		if infoset_freq * infoset_gain > 0:
			infoset_gains2.append(infoset_gain * infoset_freq)
		else:
			infoset_gains2.append(-20000.0)

	x = np.arange(len(player2_empir_infostates))
	infoset_inds_2 = None
	try:
		infoset_inds_2 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains2))
	except:
		num_nonzero = len([y for y in softmax(infoset_gains2) if y > 0.0])
		infoset_inds_2 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains2))

	player2_empir_M = [player2_empir_infostates[i] for i in infoset_inds_2]
	BR2 = convert_into_best_response_policy(player2_empir_M, policy_str, BR2_weights, game_param_map)

	BR = {}
	BR.update(BR1)
	BR.update(BR2)

	empirical_game_size_over_time.append([len(empir_strat_space), total_size(empirical_game, HANDLERS)])

	# save policy maps to disk
	np.save(prefix + "_policy_map1.npy", POLICY_SPACE1)
	np.save(prefix + "_policy_map2.npy", POLICY_SPACE2)

	while len(empirical_game_size_over_time) < NUM_PSRO_ITER and (regret_over_time[-1] > REGRET_THRESH):
		old_empir_strat_space = empir_strat_space.copy()

		# Load policy maps from disk
		POLICY_SPACE1 = np.load(prefix + "_policy_map1.npy", allow_pickle=True).item()
		POLICY_SPACE2 = np.load(prefix + "_policy_map2.npy", allow_pickle=True).item()

		total_NF_sample_budget = get_total_nf_budget(SAMPLING_BUDGET, len(empirical_game_size_over_time))
		
		# Simulate initial policy as well at start of TE-PSRO
		if len(empirical_game_size_over_time) == 1:
			payoffs, new_observations = simulate(game_param_map, empir_strat_space, {}, SAMPLING_BUDGET, STD_DEV, payoffs, POLICY_SPACE1, POLICY_SPACE2,
				default_policy1, default_policy2)
			empirical_game = mgame['game']
			empirical_game.update_game_with_simulation_output(new_observations, payoffs)
			mgame['game'] = empirical_game
		
		payoffs, new_observations = simulate(game_param_map, empir_strat_space, BR, total_NF_sample_budget, STD_DEV, payoffs, POLICY_SPACE1, POLICY_SPACE2,
			default_policy1, default_policy2)
		empirical_game = mgame['game']

		empirical_game.update_game_with_simulation_output(new_observations, payoffs)
		mgame['game'] = empirical_game
		del new_observations

		for i in range(2):
			for infoset in empirical_game.infosets[i]:
				infoset_id = infoset.infoset_id
				if infoset_id not in old_empir_strat_space.keys():
					empir_strat_space[infoset_id] = infoset.action_space[:]
				else:
					empir_strat_space[infoset_id] = infoset.action_space[:]

		empirical_game_size_over_time.append([len(empir_strat_space), total_size(empirical_game, HANDLERS)])

		if need_NE:
			ne_ms = empirical_game.cfr(T)
			if eval_strat == "NE":
				regret_meta_strat = ne_ms
			if br_mss == "NE":
				br_meta_strat = ne_ms
			ne_over_time.append(ne_ms)

		tree_of_roots, anynode_to_node_map = empirical_game.get_subgame_roots()

		if need_SPE:
			spe_ms = empirical_game.compute_SPE(T, tree_of_roots, anynode_to_node_map)
			if eval_strat == "SPE":
				regret_meta_strat = spe_ms
			if br_mss == "SPE":
				br_meta_strat = spe_ms
			ne_over_time.append(spe_ms)			
		
		max_height = tree_of_roots.height + 1
		del tree_of_roots
		del anynode_to_node_map
		del max_height

		regret = compute_regret(regret_meta_strat, eval_string, game_param_map, prefix, hp_set1, hp_set2, POLICY_SPACE1.copy(), POLICY_SPACE2.copy(), default_policy1, default_policy2)
		regret_over_time.append(regret)
		print("regret_over_time so far ", regret_over_time)

		BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(br_meta_strat, mss_string, prefix, game_param_map, hp_set1, hp_set2, 
			POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, False)

		player1_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 1]
		num_br_samples = min(NUM_EMPIR_BR, len(player1_empir_infostates))
		policy_str = "pi_" + str(len(POLICY_SPACE1))
		POLICY_SPACE1[policy_str] = BR1_weights

		infoset_gains1 = []
		for infoset_id in player1_empir_infostates:
			infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 1, BR1_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			
			if infoset_freq * infoset_gain > 0:
				infoset_gains1.append(infoset_gain * infoset_freq)
			else:
				infoset_gains1.append(-20000.0)

		x = np.arange(len(player1_empir_infostates))
		infoset_inds_1 = None
		try:
			infoset_inds_1 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains1))
		except:
			num_nonzero = len([y for y in softmax(infoset_gains1) if y > 0.0])
			infoset_inds_1 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains1))
			
		player1_empir_M = [player1_empir_infostates[i] for i in infoset_inds_1]
		BR1 = convert_into_best_response_policy(player1_empir_M, policy_str, BR1_weights, game_param_map)

		infoset_gains2 = []
		player2_empir_infostates = [infoset_id for infoset_id in br_meta_strat if infoset_id[0] == 2]
		num_br_samples = min(NUM_EMPIR_BR, len(player2_empir_infostates))
		policy_str = "pi_" + str(len(POLICY_SPACE2))
		POLICY_SPACE2[policy_str] = BR2_weights

		for infoset_id in player2_empir_infostates:
			infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, 2, BR2_weights, game_param_map, POLICY_SPACE1, POLICY_SPACE2)
			if infoset_freq * infoset_gain > 0:
				infoset_gains2.append(infoset_gain * infoset_freq)
			else:
				infoset_gains2.append(-20000.0)
			
		x = np.arange(len(player2_empir_infostates))
		infoset_inds_2 = None
		try:
			infoset_inds_2 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax(infoset_gains2))
		except:
			num_nonzero = len([y for y in softmax(infoset_gains2) if y > 0.0])
			infoset_inds_2 = np.random.choice(x, size=num_nonzero, replace=False, p=softmax(infoset_gains2))

		player2_empir_M = [player2_empir_infostates[i] for i in infoset_inds_2]
		BR2 = convert_into_best_response_policy(player2_empir_M, policy_str, BR2_weights, game_param_map)
		
		BR = {}
		BR.update(BR1)
		BR.update(BR2)

		# save policy maps to disk
		np.save(prefix + "_policy_map1.npy", POLICY_SPACE1)
		np.save(prefix + "_policy_map2.npy", POLICY_SPACE2)

		np.savez_compressed(prefix, regret_over_time, max_subgame_regret_over_time, empirical_game_size_over_time, ne_over_time, spe_over_time)

	del mgame['game']
	mgame.vacuum()

file_ID_index = int(sys.argv[1]) // 9
trial_index = int(sys.argv[1]) % 3
br_index = int(sys.argv[2])
NUM_EMPIR_BR = emp_br_list[br_index]

rounds_index = int(sys.argv[3])
NUM_ROUNDS = num_rounds_list[rounds_index]
included_rounds_index = int(sys.argv[1]) // 15
included_rounds = [i for i in range(included_rounds_index + 1)]

game_params = retrieve_game(file_ID_list.get(NUM_ROUNDS)[file_ID_index], NUM_ROUNDS)

game_param_map = {
	"file_ID": game_params[0],
	"num_rounds": game_params[1],
	"p1_actions": game_params[2],
	"p2_actions": game_params[3],
	"chance_events": game_params[4],
	"card_weights": game_params[5],
	"payoff_map": game_params[6],
	"included_rounds": included_rounds
}

te_egta(game_param_map, T, trial_index, br_mss="SPE", eval_strat="SPE")
