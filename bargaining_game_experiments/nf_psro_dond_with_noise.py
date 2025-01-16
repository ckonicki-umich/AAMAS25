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
import pygambit
import gambit_conversion as gbc
from Node import *
from Infoset import *
from ExtensiveForm import *
from DQN import *
from bargaining import *
from best_response_NF import *

N = 2
SAMPLING_BUDGET = 100
STD_DEV = 0.5
REGRET_THRESH = 0.1
NUM_PSRO_ITER = 30
NUM_PLAYER_TURNS = 5
NUM_ITEM_TYPES = 3
MAX_POOL_SIZE = 7
MIN_POOL_SIZE = 5
VAL_TOTAL = 10
OUTSIDE_OFFERS = ["H", "L"]

file_ID_list = [
'BIG_DoND_161GZ', 
'BIG_DoND_BECPD', 
'BIG_DoND_87YP1', 
'BIG_DoND_GTZE3', 
'BIG_DoND_ASNW5', 
'BIG_DoND_XI2O1', 
'BIG_DoND_4MSNK', 
'BIG_DoND_YRZVS', 
'BIG_DoND_OVL2Y',
'BIG_DoND_RBK4B', 
'BIG_DoND_SZV54', 
'BIG_DoND_9H7NI', 
'BIG_DoND_RBINZ', 
'BIG_DoND_JW5X5'
]

def retrieve_game(file_ID_index):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	a_f = np.load("game_parameters.npz", allow_pickle=True)
	lst = a_f.files
	for params in a_f['arr_0']:
		if file_ID_index == params[0]:
			return params

def retrieve_json_hps(file_ID_index, player_num):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration
	@arg (int) player_num: index {1, 2} corresponding to one of the two players

	Helper method to retrieve each set of learned hyperparameter values from
	phases 1 and 2 for a given game and player
	'''
	with open('optimal_learned_hp_DoND.json') as f:
		data = f.read()
	js = json.loads(data)

	d_both = js[str(file_ID_index)]
	d = d_both[str(player_num)]
	keys = d.keys()
	hp_set = None
	for elt in it.product(*d.values()):

		hp_set = dict(zip(keys, elt))

	return hp_set

def simulate(val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, pool, empir_strat1, empir_strat2, num_iter, noise, payoffs, 
	NF_POLICY_SPACE1, NF_POLICY_SPACE2):
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
	for n in range(num_iter):
		if n % 1000 == 0:
			print("n ", n, "of ", num_iter)

		v1, v2 = generate_player_valuations(val_dist)
		o1, o2 = generate_player_outside_offers(outside_offer_dist1, outside_offer_dist2)
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

		current_history = [v1, o1, v2, o2]
		num_prev_rounds = 0

		for turn in range(NUM_PLAYER_TURNS):
			# player 1 makes initial offer to player 2
			offer_space1 = generate_offer_space(pool)
			action_space = list(it.product(offer_space1, [True, False]))

			if turn == 0:
				offer_space1.remove(('deal',))
				offer_space1.remove(('walk',))

			p1_signal = empir_strat1[1]
			p1_offer = get_offer_given_nf_policy(empir_strat1[0], action_space, NF_POLICY_SPACE1, tuple(current_history), pool)
			p1_action = (p1_offer, p1_signal)			
			current_history += [p1_action]
			
			if p1_action[0] in [("walk",), ("deal",)]:
				break

			# player 2 makes counteroffer to player 1
			p2_signal = empir_strat2[1]
			p2_offer = get_offer_given_nf_policy(empir_strat2[0], action_space, NF_POLICY_SPACE2, tuple(current_history), pool)
			p2_action = (p2_offer, p2_signal)
			current_history += [p2_action]
			if p2_action[0] == ("walk",) or p2_action[0] == ("deal",):
				break

			num_prev_rounds += 1
			
		utility = None
		split = current_history[-2][0]
		if current_history[-1] == ("deal",):
			utility = compute_utility(("deal",), pool, v1, v2, split, o1_pay, o2_pay, num_prev_rounds)
		else:
			utility = compute_utility(("walk",), pool, v1, v2, split, o1_pay, o2_pay, num_prev_rounds)

		payoff_sample = np.random.normal(utility, np.array([noise] * 2))
		payoffs[(empir_strat1, empir_strat2)] = payoffs.get((empir_strat1, empir_strat2), []) + [payoff_sample]

	return payoffs

def compute_true_pay_NF(meta_strategy1, meta_strategy2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2):
	'''
	@arg (map) meta_strategy: given strategy profile
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")

	Computes the payoff of playing a given strategy profile in the true game "tree"
	'''
	pay = np.zeros(N)
	for nf_strat1 in meta_strategy1:
		strat_weight1 = meta_strategy1.get(nf_strat1)
		for nf_strat2 in meta_strategy2:
			strat_weight2 = meta_strategy2.get(nf_strat2)
			strat_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = recursive_true_pay_helper_NF([], nf_strat1, nf_strat2, 1.0, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
				NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			pay += strat_weight1 * strat_weight2 * strat_pay

	return pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2


def recursive_true_pay_helper_NF(action_history, player1_strategy, player2_strategy, input_reach_prob, pool, val_dist, 
	outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
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
	turns_taken = len(action_history) - 4
	num_prev_rounds = turns_taken // 2

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(action_history) == 0:
		pay = np.zeros(N)
		#'''
		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper_NF(next_node, player1_strategy, player2_strategy, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
		
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		pay = np.zeros(N)	
		for o1 in OUTSIDE_OFFERS:
			prob = 0.5
			next_node = action_history + (o1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper_NF(next_node, player1_strategy, player2_strategy, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# third chance node has been reached -- valuation for player 2 in true game
	# elif len(history) == 1:
	elif len(action_history) == 2:
		pay = np.zeros(N)
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = action_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper_NF(next_node, player1_strategy, player2_strategy, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(action_history) == 3:
		pay = np.zeros(N)
		for o2 in OUTSIDE_OFFERS:
			prob = 0.5
			next_node = action_history + (o2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper_NF(next_node, player1_strategy, player2_strategy, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	elif action_history[-1][0] in [("walk",), ("deal",)] or turns_taken >= NUM_PLAYER_TURNS * 2:

		is_deal = action_history[-1][0]
		if is_deal not in [("walk",), ("deal",)]:
			is_deal = ("walk",)

		v1 = action_history[0]
		o1 = action_history[1]
		v2 = action_history[2]
		o2 = action_history[3]
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
		util_vec = compute_utility(is_deal, pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_prev_rounds)
		
		return util_vec * input_reach_prob, POLICY_SPACE1, POLICY_SPACE2
	else:
		pay = np.zeros(N)
		player_num = 1
		strategy = player1_strategy
		player_history = get_player_history_given_action_history(action_history)
		PS = POLICY_SPACE1.copy()

		if len(action_history) % 2 == 1:
			player_num = 2
			PS = POLICY_SPACE2.copy()
			strategy = player2_strategy

		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))
		policy_str = strategy[0]
		offer = get_offer_given_nf_policy(policy_str, action_space, PS, action_history, pool)
		a = (offer, strategy[1])
		next_node = action_history + (a,)
		next_reach_prob = input_reach_prob
		new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper_NF(next_node, player1_strategy, player2_strategy, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
		pay = pay + new_pay

	return pay, POLICY_SPACE1, POLICY_SPACE2

def compute_nf_br_weights_pay_helper(action_history, BR1_weights, BR2_weights, br_player, input_reach_prob, nf_strat1, nf_strat2, pool, val_dist, 
	outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2):
	'''
	'''
	turns_taken = len(action_history) - 4
	num_prev_rounds = turns_taken // 2

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(action_history) == 0:
		pay = 0.0
		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper(next_node, BR1_weights, BR2_weights, br_player, next_reach_prob,
				nf_strat1, nf_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			pay = pay + new_pay

	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		pay = 0.0
		for o1 in OUTSIDE_OFFERS:
			prob = outside_offer_dist1.get(o1)
			next_node = action_history + (o1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper(next_node, BR1_weights, BR2_weights, br_player, next_reach_prob,
				nf_strat1, nf_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			pay = pay + new_pay	

	# third chance node has been reached -- valuation for player 2 in true game
	elif len(action_history) == 2:
		pay = 0.0
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = action_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper(next_node, BR1_weights, BR2_weights, br_player, next_reach_prob,
				nf_strat1, nf_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			pay = pay + new_pay

	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(action_history) == 3:
		pay = 0.0
		for o2 in OUTSIDE_OFFERS:
			prob = outside_offer_dist2.get(o2)
			next_node = action_history + (o2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper(next_node, BR1_weights, BR2_weights, br_player, next_reach_prob,
				nf_strat1, nf_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			pay = pay + new_pay

	elif action_history[-1][0] in [("walk",), ("deal",)] or turns_taken >= NUM_PLAYER_TURNS * 2:

		is_deal = action_history[-1][0]
		if is_deal not in [("walk",), ("deal",)]:
			is_deal = ("walk",)

		v1 = action_history[0]
		o1 = action_history[1]
		v2 = action_history[2]
		o2 = action_history[3]
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
		util_vec = compute_utility(is_deal, pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_prev_rounds)
		util = util_vec[br_player - 1]
		
		return util * input_reach_prob, NF_POLICY_SPACE1, NF_POLICY_SPACE2

	else:
		pay = 0.0
		player_num = 1
		PS = NF_POLICY_SPACE1.copy()
		BR_network_weights = BR1_weights
		nf_strat = nf_strat1
		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))

		if len(action_history) % 2 == 1:
			player_num = 2
			PS = NF_POLICY_SPACE2.copy()
			BR_network_weights = BR2_weights
			nf_strat = nf_strat2

		if player_num != br_player:

			offer = get_offer_given_nf_policy(nf_strat[0], action_space, NF_POLICY_SPACE2, action_history, pool)
			next_node = action_history + ((offer, nf_strat[1]),)
			next_reach_prob = input_reach_prob
			new_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper(next_node, BR1_weights, BR2_weights, br_player, next_reach_prob,
				nf_strat1, nf_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			pay = pay + new_pay
		else:
			state = convert_into_state(action_history, pool)
			best_action = get_best_action(state, BR_network_weights, action_space, len(action_history) <= 4)
			next_node = action_history + (best_action,)
			next_reach_prob = input_reach_prob
			new_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper(next_node, BR1_weights, BR2_weights, br_player, next_reach_prob,
				nf_strat1, nf_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			pay = pay + new_pay	

	return pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2

def compute_regret_NF(ms1, ms2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
	file_ID, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2):
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
	meta_strategy_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_true_pay_NF(ms1, ms2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
		NF_POLICY_SPACE1, NF_POLICY_SPACE2)
	regrets = []
	BR1_weights, BR2_weights, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_best_response_NF(ms1, ms2, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
		o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2, True)
	
	action_pay1 = 0.0
	for nf_strat2 in ms2:
		strat_weight2 = ms2.get(nf_strat2)
		strat_pay1, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper([], BR1_weights, None, 1, 1.0, None, nf_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
			o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
		action_pay1 += strat_weight2 * strat_pay1

	regrets.append(max(action_pay1 - meta_strategy_pay[0], 0.0))

	action_pay2 = 0.0
	for nf_strat1 in ms1:
		strat_weight1 = ms1.get(nf_strat1)
		strat_pay2, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_nf_br_weights_pay_helper([], None, BR2_weights, 2, 1.0, nf_strat1, None, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
			o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
		action_pay2 += strat_weight1 * strat_pay2

	regrets.append(max(action_pay2 - meta_strategy_pay[1], 0.0))

	return max(regrets)

def construct_initial_NF_policy(game_param_map, hp_set):
	'''
	'''
	signals = ["H", "L"]
	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	offer_space = generate_offer_space(pool)
	offer_space1 = offer_space[:]
	offer_space1.remove(("deal",))
	offer_space1.remove(("walk",))

	initial_policy1 = {}
	initial_policy2 = {}
	NF_POLICY_SPACE1 = {}
	NF_POLICY_SPACE2 = {}

	state_len = get_state_length(pool)
	action_space = list(it.product(offer_space, [True, False]))

	default_model1 = tf.keras.Sequential()
	default_model1.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_model1.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))
	default_model1.add(tf.keras.layers.Dense(len(action_space)))
	default_model1.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))
	W1_default = default_model1.get_weights()

	default_model2 = tf.keras.Sequential()
	default_model2.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_model2.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))
	default_model2.add(tf.keras.layers.Dense(len(action_space)))
	default_model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))
	W2_default = default_model2.get_weights()

	reveal1, reveal2 = np.random.choice([True, False], 2)
	initial_policy1 = {("pi_0", reveal1): 1.0}
	initial_policy2 = {("pi_0", reveal2): 1.0}
	NF_POLICY_SPACE1["pi_0"] = W1_default
	NF_POLICY_SPACE2["pi_0"] = W2_default

	return initial_policy1, initial_policy2, NF_POLICY_SPACE1, NF_POLICY_SPACE2

def nf_egta(game_param_map, trial_index):
	'''
	@arg (map) initial_sigma: Initial metastrategy based on empirical strategy
		space
	@arg (map) game_param_map: map of game parameters for given file ID (player valuation distribution, item pool,
		outside offer distributions)
	@arg (int) T: Number of iterations for a single run of CFR (whether solving a game for NE or a subgame as part
		of solving a game for SPE) -- probably should remove this

	Runs a single play of NF-EGTA on large sequential bargaining game, expanding strategy space, simulating
	each new strategy and constructing the empirical game matrix, which is then solved for an
	approximate NE
	'''
	file_ID = game_param_map["file_ID"]
	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	#extract learned hyperparameters for DQN
	hp_set1 = retrieve_json_hps(file_ID_index, 1)
	hp_set2 = retrieve_json_hps(file_ID_index, 2)

	# Initialize the empirical strategy space based on initial_sigma
	initial_sigma1, initial_sigma2, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = construct_initial_NF_policy(game_param_map, hp_set1)
	empir_strat_space1 = [list(initial_sigma1.keys())[0]]
	empir_strat_space2 = [list(initial_sigma2.keys())[0]]

	file_prefix = 'BIG_DoND_NF-PSRO_noise_' + file_ID + "_" + str(trial_index)
	file_name = file_prefix + '_empirical_game.mmdpickle'
	with open(file_name, 'w') as fp:
		pass

	mgame = mmp.mmapdict(file_name)
	empirical_game_matrix = {}	
	mgame['game'] = empirical_game_matrix

	regret_over_time = []
	payoff_estimation_error_over_time = []
	br1_payoffs_over_time = []
	br2_payoffs_over_time = []
	payoffs = {}

	ne_meta_strat1 = initial_sigma1.copy()
	ne_meta_strat2 = initial_sigma2.copy()
	empirical_game_size_over_time = []
	solution_over_time = []

	regret = compute_regret_NF(ne_meta_strat1, ne_meta_strat2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, file_prefix, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2)

	# BR1, BR2, br1_pay, br2_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_best_response_NF(ne_meta_strat1, ne_meta_strat2, file_prefix, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
	# 	o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
	BR1, BR2, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_best_response_NF(ne_meta_strat1, ne_meta_strat2, file_prefix, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2, False)

	regret_over_time.append(regret)
	print("regret_over_time so far ", regret_over_time)
	empirical_game_size_over_time.append(len(empir_strat_space1) * len(empir_strat_space2))
	# br1_payoffs_over_time.append(br1_pay)
	# br2_payoffs_over_time.append(br2_pay)

	# save policy maps to disk
	np.save(file_prefix + "_policy_map1.npy", NF_POLICY_SPACE1)
	np.save(file_prefix + "_policy_map2.npy", NF_POLICY_SPACE2)

	while len(regret_over_time) < NUM_PSRO_ITER and regret_over_time[-1] > REGRET_THRESH:
		if BR1 not in empir_strat_space1:
			empir_strat_space1.append(BR1)

		if BR2 not in empir_strat_space2:
			empir_strat_space2.append(BR2)

		empirical_game_size_over_time.append(len(empir_strat_space1) * len(empir_strat_space2))
		gc.collect()

		NF_POLICY_SPACE1 = np.load(file_prefix + "_policy_map1.npy", allow_pickle=True).item()
		NF_POLICY_SPACE2 = np.load(file_prefix + "_policy_map2.npy", allow_pickle=True).item()

		print("simulating true game and updating empirical game w/ new simulation data")
		num_iter1 = SAMPLING_BUDGET
		num_iter2 = SAMPLING_BUDGET
		empirical_game_matrix = mgame['game']
		print(empirical_game_matrix)

		for empir_strat1 in empir_strat_space1:
			for empir_strat2 in empir_strat_space2:
				if (empir_strat1, empir_strat2) not in empirical_game_matrix:
					payoffs = simulate(val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
						pool, empir_strat1, empir_strat2, num_iter1, STD_DEV, payoffs, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
					pay = payoffs[(empir_strat1, empir_strat2)]
					empirical_game_matrix[(empir_strat1, empir_strat2)] = np.mean(pay, axis=0)

		# Update empirical payoff matrix with new simulation data
		mgame['game'] = empirical_game_matrix
		num_combs = 0
		avg_err = 0

		for policy_str1 in empir_strat_space1:
			empir_strat1 = {policy_str1: 1.0}
			for policy_str2 in empir_strat_space2:
				empir_strat2 = {policy_str2: 1.0}
				est_pay = empirical_game_matrix[(policy_str1, policy_str2)]
				true_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_true_pay_NF(empir_strat1, empir_strat2, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2)

				est_err = max(abs(est_pay - true_pay))
				avg_err += est_err
				num_combs += 1

				if num_combs == 500:
					break
		
		avg_err /= num_combs
		payoff_estimation_error_over_time.append(avg_err)

		print("computing new metastrategy")
		gbc.write_nfg_file(empirical_game_matrix, empir_strat_space1, empir_strat_space2, file_prefix)
		g = pygambit.Game.read_game("empirical_game_" + file_prefix + ".nfg")
		ne = pygambit.nash.lcp_solve(g, use_strategic=True)[0]
		ne_1 = [float(x) for x in ne[g.players[0]]]
		ne_2 = [float(x) for x in ne[g.players[0]]]

		ms1 = {}
		ms2 = {}
		for i in range(len(empir_strat_space1)):
			policy_str1 = empir_strat_space1[i]
			ms1[policy_str1] = ne_1[i]

		for i in range(len(empir_strat_space2)):
			policy_str2 = empir_strat_space2[i]
			ms2[policy_str2] = ne_2[i]

		print("ms1 ", ms1)
		print("ms2 ", ms2)
		solution_over_time.append((ms1, ms2))

		print("computing regret with new metastrategy")
		# regret, BR1, BR2, br1_pay, br2_pay, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_regret_NF(ms1, ms2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		# 	o1_pay_arr, o2_pay_arr, file_prefix, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
		regret = compute_regret_NF(ms1, ms2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, file_prefix, hp_set1, hp_set2, 
			NF_POLICY_SPACE1, NF_POLICY_SPACE2)

		BR1, BR2, NF_POLICY_SPACE1, NF_POLICY_SPACE2 = compute_best_response_NF(ms1, ms2, file_prefix, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2, False)

		regret_over_time.append(regret)
		print("regret_over_time so far ", regret_over_time)
		# br1_payoffs_over_time.append(br1_pay)
		# br2_payoffs_over_time.append(br2_pay)

		# save policy maps to disk
		np.save(file_prefix + "_policy_map1.npy", NF_POLICY_SPACE1)
		np.save(file_prefix + "_policy_map2.npy", NF_POLICY_SPACE2)

		# np.savez_compressed(file_prefix, regret_over_time, empirical_game_size_over_time, payoff_estimation_error_over_time, 
		# 	br1_payoffs_over_time, br2_payoffs_over_time)
		np.savez_compressed(file_prefix, regret_over_time, empirical_game_size_over_time, payoff_estimation_error_over_time, 
			solution_over_time)

	del mgame['game']
	mgame.vacuum()


file_ID_index = int(sys.argv[1]) // 5
trial_index = int(sys.argv[1]) % 5
file_ID = file_ID_list[file_ID_index]
game_params = retrieve_game(file_ID_list[file_ID_index])

game_param_map = {
	"file_ID": game_params[0],
	"pool": game_params[1], 
	"val_dist": game_params[2],
	"ood1": game_params[3],
	"ood2": game_params[4],
	"o1_pay": game_params[5],
	"o2_pay": game_params[6]
}

nf_egta(game_param_map, trial_index)

