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
from bargaining import *
from best_response import *
from compute_memory import *

N = 2
SAMPLING_BUDGET = 100
#SAMPLING_BUDGET = 5000
#SAMPLING_BUDGET = 100
#SAMPLING_BUDGET = 50
#STD_DEV = 0.2
STD_DEV = 0.5
#T = 1000
REGRET_THRESH = 0.1
# NUM_PSRO_ITER = 30
NUM_PSRO_ITER = 20

T = 500

# Original game: 5 turns per player
NUM_PLAYER_TURNS = 3
NUM_ITEM_TYPES = 3
MAX_POOL_SIZE = 7
MIN_POOL_SIZE = 5
VAL_TOTAL = 10
OUTSIDE_OFFERS = ["H", "L"]

HANDLERS = {
ExtensiveForm: ExtensiveFormHandler,
Infoset: InfosetHandler,
Node: NodeHandler}

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

def retrieve_game(file_ID_index):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	a_f = np.load("game_parameters.npz", allow_pickle=True)
	lst = a_f.files
	for params in a_f['arr_0']:
		#print("params ", params)
		#print("\n")
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
	#hp_set_list = []
	#count = 0
	hp_set = None
	for elt in it.product(*d.values()):

		hp_set = dict(zip(keys, elt))

	#print("hp_set ", hp_set)
	
	return hp_set

def generate_new_BR_paths(empir_strat_space, BR):
	'''
	'''
	# len(BR) = 2 * M

	NUM_BR = 2**len(BR)
	print("NUM_BR ", NUM_BR)

	# get every possible combination of BR's to be included in tree
	for j in range(1, NUM_BR):
		print("j ", j)
		strat = {}

		bin_list = [int(x) for x in bin(j)[2:]]
		#print("bin_list_j ", bin_list)

		if len(bin_list) < len(BR):
			bin_list_copy = bin_list[:]
			bin_list = [0] * (len(BR) - len(bin_list_copy)) + bin_list_copy

			#print("updated ", bin_list)

		br_list = list(it.compress(BR, bin_list))
		print(br_list)

		for infoset_id in empir_strat_space:
			if infoset_id in br_list:
				strat[infoset_id] = [BR[infoset_id]]
			else:
				strat[infoset_id] = empir_strat_space[infoset_id][:]


		print("strat ", strat)
		yield strat

def get_total_nf_budget(SAMPLING_BUDGET, complete_psro_iter):
	'''
	'''
	num_cells_square = complete_psro_iter**2
	print("num_cells_square ", num_cells_square)
	num_new_cells_square = (complete_psro_iter + 1)**2
	print("num_new_cells_square ", num_new_cells_square)

	return (num_new_cells_square - num_cells_square) * SAMPLING_BUDGET

def simulate(val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, pool, old_strategy_space, BR, total_NF_sample_budget, noise, payoffs, 
	POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
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
	# observations = []
	observations = {}
	strategy_space = old_strategy_space.copy()

	num_tree_paths = max(2**len(BR) - 1, 1)
	print("num_tree_paths ", num_tree_paths)
	num_iter = int(float(total_NF_sample_budget) / num_tree_paths)
	print("num_iter ", num_iter)

	for strategy in generate_new_BR_paths(old_strategy_space, BR):
		#print("strategy ", strategy)

		for n in range(num_iter):
			if n % 1000 == 0:
				print("n ", n, "of ", num_iter)

			v1, v2 = generate_player_valuations(val_dist)
			o1, o2 = generate_player_outside_offers(outside_offer_dist1, outside_offer_dist2)

			o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
			o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

			action_history = [v1, o1, v2, o2]
			empir_history = [o1, o2]

			event_index = None
			next_node = None

			utility = None

			reached_end = False

			for turn in range(NUM_PLAYER_TURNS):
				#print("reached_end ", reached_end)
				empir_infoset_id1 = get_empirical_infoset_id_given_histories(tuple(action_history), pool, POLICY_SPACE1, POLICY_SPACE2)
				#print("empir_infoset_id1 ", empir_infoset_id1)
				num_rounds = (len(action_history) - 4) // 2

				# player 1 makes initial offer to player 2
				offer_space1 = generate_offer_space(pool)
				action_space = list(it.product(offer_space1, [True, False]))

				# don't allow the action taken or the policy chosen to advise player 1 to accept a deal if nothing has happened
				if turn == 0:
					offer_space1.remove(('deal',))
					offer_space1.remove(('walk',))

				p1_empir_action = None
				if empir_infoset_id1 not in strategy:
					#print("not1")
					p1_empir_action = empir_history[-2]
					p1_empir_action = (p1_empir_action[0], bool(p1_empir_action[1]))
					#print("previous_empir_action1 ", p1_empir_action)
					#print(reached_end)
					if not reached_end:
						reached_end = True
						empir_history += [p1_empir_action]
						#strategy_space[empir_infoset_id1] = [p1_empir_action]
					
				else:
					p1_empir_action = random.choice(strategy.get(empir_infoset_id1))
					p1_empir_action = (p1_empir_action[0], bool(p1_empir_action[1]))
					#print("p1_empir_action ", p1_empir_action)

					empir_history += [p1_empir_action]

				#print("empir_history ", empir_history)

				p1_policy_str = p1_empir_action[0]
				
				p1_offer = get_offer_given_policy(p1_policy_str, action_space, POLICY_SPACE1, tuple(action_history), pool)
				p1_action = (p1_offer, p1_empir_action[1])

				#print("p1_offer ", p1_offer)
				
				action_history += [p1_action]
				#print("action_history ", action_history)


				# check for walking or deal
				if p1_action[0] in [("walk",), ("deal",)]:
					empir_history += [p1_empir_action]

					if p1_action[0] == ("walk",):
						utility = compute_utility(("walk",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)
					else:
						utility = compute_utility(("deal",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)

					break

				# player 2 makes counteroffer to player 1
				empir_infoset_id2 = get_empirical_infoset_id_given_histories(tuple(action_history), pool, POLICY_SPACE1, POLICY_SPACE2)
				#print("empir_infoset_id2 ", empir_infoset_id2)
				num_rounds = (len(action_history) - 4) // 2
				offer_space2 = generate_offer_space(pool)
				action_space = list(it.product(offer_space2, [True, False]))

				p2_empir_action = None
				if empir_infoset_id2 not in strategy:
					#print("not2")
					if len(empir_history) > 4:
						p2_empir_action = empir_history[-2]
					else:
						p2_empir_action = default_policy2
						#print("p2_empir_action ", p2_empir_action)

					p2_empir_action = (p2_empir_action[0], bool(p2_empir_action[1]))

					#print("previous_empir_action2 ", p2_empir_action)
					#print(reached_end)
					if not reached_end:
						reached_end = True
						empir_history += [p2_empir_action]
						#strategy_space[empir_infoset_id2] = [p2_empir_action]

				else:
					p2_empir_action = random.choice(strategy.get(empir_infoset_id2))
					p2_empir_action = (p2_empir_action[0], bool(p2_empir_action[1]))
					empir_history += [p2_empir_action]
					#print("p2_empir_action ", p2_empir_action)

				
				p2_offer = get_offer_given_policy(p2_empir_action[0], action_space, POLICY_SPACE2, tuple(action_history), pool)
				p2_action = (p2_offer, p2_empir_action[1])

				action_history += [p2_action]
				#print("empir_history ", empir_history)

				# check for walking or deal
				if p2_action[0] in [("walk",), ("deal",)]:

					if p2_action[0] == ("walk",):
						utility = compute_utility(("walk",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)
					else:				
						utility = compute_utility(("deal",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_rounds)

					break

			# if empir_history not in observations:
			# 	observations.append(empir_history)
			# 	print("empir_history ", empir_history)
			#observations.append(empir_history)
			observations[tuple(empir_history)] = observations.get(tuple(empir_history), 0.0) + 1
			#print("empir_history ", empir_history)

			# ran out of time to make a deal/end negotiations --> number of rounds elapsed needs to equal
			# the number of player turns
			if utility is None:
				utility = compute_utility(("walk",), pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, NUM_PLAYER_TURNS)

			payoff_sample = np.random.normal(utility, np.array([noise] * 2))
			#print("payoff_sample ", payoff_sample)
			payoffs[tuple(empir_history)] = payoffs.get(tuple(empir_history), []) + [payoff_sample]

	# return payoffs, observations, strategy_space
	return payoffs, observations

def compute_true_pay(empirical_strategy_profile, BR1_weights, BR2_weights, j, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
	o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	return recursive_true_pay_helper([], empirical_strategy_profile, BR1_weights, BR2_weights, j, 1.0, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, None, None, POLICY_SPACE1, POLICY_SPACE2)

def recursive_true_pay_helper(action_history, strategy_profile, BR1_weights, BR2_weights, br_player, input_reach_prob, pool, val_dist, 
	outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	turns_taken = len(action_history) - 4
	num_prev_rounds = turns_taken // 2

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(action_history) == 0:
		pay = np.zeros(N)

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
		
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		pay = np.zeros(N)	
		for o1 in OUTSIDE_OFFERS:
			#prob = 0.5
			prob = outside_offer_dist1.get(o1)
			next_node = action_history + (o1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# third chance node has been reached -- valuation for player 2 in true game
	elif len(action_history) == 2:
		pay = np.zeros(N)
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = action_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player,
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(action_history) == 3:
		pay = np.zeros(N)
		for o2 in OUTSIDE_OFFERS:
			#prob = 0.5
			prob = outside_offer_dist2.get(o2)
			next_node = action_history + (o2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
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
		PS = POLICY_SPACE1.copy()
		BR_network_weights = BR1_weights
		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))

		if len(action_history) % 2 == 1:
			player_num = 2
			PS = POLICY_SPACE2.copy()
			BR_network_weights = BR2_weights

		if player_num != br_player:
			empir_infoset_id = get_empirical_infoset_id_given_histories(action_history, pool, POLICY_SPACE1, POLICY_SPACE2)

			assert empir_infoset_id[0] == player_num

			infoset_strat = strategy_profile.get(empir_infoset_id)
			game_start = len(action_history) <= 4

			if infoset_strat is not None:
				for empir_action in infoset_strat.keys():
					policy_str = empir_action[0]
					if infoset_strat.get(empir_action) > 0.0:
						offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
						next_node = action_history + ((offer, empir_action[1]),)
						next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)

						new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
							next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
						pay = pay + new_pay

			else:
				signal = random.choice([True, False])
				
				if len(action_history) <= 5:
					policy_str = "pi_0"
				else:
					last_policy_str = empir_infoset_id[1][-2][0]
					#print("hey")
					if last_policy_str[0] is None:
						last_policy_str = "pi_0"

					policy_str = last_policy_str

				empir_action = (policy_str, signal)
				offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
				a = (offer, signal)
				next_node = action_history + (a,)
				next_reach_prob = input_reach_prob

				new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
					next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay

		else:
			o1 = action_history[1]
			o2 = action_history[3]

			state = convert_into_state(action_history, pool)

			best_action = get_best_action(state, BR_network_weights, action_space, len(action_history) <= 4)

			next_node = action_history + (best_action,)
			next_reach_prob = input_reach_prob

			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

			if best_action is None:
				print(nah)

	return pay, POLICY_SPACE1, POLICY_SPACE2

def compute_true_pay(empirical_strategy_profile, BR1_weights, BR2_weights, j, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
	o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	return recursive_true_pay_helper([], empirical_strategy_profile, BR1_weights, BR2_weights, j, 1.0, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, None, None, POLICY_SPACE1, POLICY_SPACE2)

def recursive_true_pay_helper(action_history, strategy_profile, BR1_weights, BR2_weights, br_player, input_reach_prob, pool, val_dist, 
	outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	turns_taken = len(action_history) - 4
	num_prev_rounds = turns_taken // 2

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(action_history) == 0:
		pay = np.zeros(N)

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, v1, None, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
		
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		pay = np.zeros(N)	
		for o1 in OUTSIDE_OFFERS:
			#prob = 0.5
			prob = outside_offer_dist1.get(o1)
			next_node = action_history + (o1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, v1, None, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# third chance node has been reached -- valuation for player 2 in true game
	elif len(action_history) == 2:
		pay = np.zeros(N)
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = action_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player,
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(action_history) == 3:
		pay = np.zeros(N)
		for o2 in OUTSIDE_OFFERS:
			#prob = 0.5
			prob = outside_offer_dist2.get(o2)
			next_node = action_history + (o2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	elif action_history[-1][0] in [("walk",), ("deal",)] or turns_taken >= NUM_PLAYER_TURNS * 2:

		is_deal = action_history[-1][0]
		if is_deal not in [("walk",), ("deal",)]:
			is_deal = ("walk",)

		o1 = action_history[1]
		o2 = action_history[3]
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
		util_vec = compute_utility(is_deal, pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_prev_rounds)

		return util_vec * input_reach_prob, POLICY_SPACE1, POLICY_SPACE2

	else:
		pay = np.zeros(N)
		player_num = 1
		v = v1
		PS = POLICY_SPACE1.copy()
		BR_network_weights = BR1_weights
		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))

		if len(action_history) % 2 == 1:
			player_num = 2
			v = v2
			PS = POLICY_SPACE2.copy()
			BR_network_weights = BR2_weights

		if player_num != br_player:
			empir_infoset_id = get_empirical_infoset_id_given_histories(action_history, pool, POLICY_SPACE1, POLICY_SPACE2)

			assert empir_infoset_id[0] == player_num

			infoset_strat = strategy_profile.get(empir_infoset_id)
			game_start = len(action_history) <= 4

			if infoset_strat is not None:
				for empir_action in infoset_strat.keys():
					policy_str = empir_action[0]
					if infoset_strat.get(empir_action) > 0.0:
						offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
						next_node = action_history + ((offer, empir_action[1]),)
						next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)

						new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
						pay = pay + new_pay

			else:
				signal = random.choice([True, False])
				
				if len(action_history) <= 5:
					policy_str = "pi_0"
				else:
					last_policy_str = empir_infoset_id[1][-2][0]
					#print("hey")
					if last_policy_str[0] is None:
						last_policy_str = "pi_0"

					policy_str = last_policy_str

				empir_action = (policy_str, signal)
				offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
				a = (offer, signal)
				next_node = action_history + (a,)
				next_reach_prob = input_reach_prob

				new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay

		else:
			o1 = action_history[1]
			o2 = action_history[3]

			state = convert_into_state(action_history, pool)

			best_action = get_best_action(state, BR_network_weights, action_space, len(action_history) <= 4)

			next_node = action_history + (best_action,)
			next_reach_prob = input_reach_prob

			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

			if best_action is None:
				print(nah)

	return pay, POLICY_SPACE1, POLICY_SPACE2

def compute_true_empirical_strategy_pay(meta_strategy, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, 
	o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
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
	return recursive_true_empirical_strategy_pay_helper([], meta_strategy, 1.0, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, None, None, POLICY_SPACE1, POLICY_SPACE2)

def recursive_true_empirical_strategy_pay_helper(action_history, strategy_profile, input_reach_prob, pool, val_dist, 
	outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2):
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

	# print("action_history ", action_history)

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(action_history) == 0:
		pay = np.zeros(N)

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, None, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
		
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		pay = np.zeros(N)	
		for o1 in OUTSIDE_OFFERS:
			#prob = 0.5
			prob = outside_offer_dist1.get(o1)
			next_node = action_history + (o1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, None, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# third chance node has been reached -- valuation for player 2 in true game
	elif len(action_history) == 2:
		pay = np.zeros(N)
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = action_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(action_history) == 3:
		pay = np.zeros(N)
		for o2 in OUTSIDE_OFFERS:
			#prob = 0.5
			prob = outside_offer_dist2.get(o2)
			next_node = action_history + (o2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	elif action_history[-1][0] in [("walk",), ("deal",)] or turns_taken >= NUM_PLAYER_TURNS * 2:

		is_deal = action_history[-1][0]
		if is_deal not in [("walk",), ("deal",)]:
			is_deal = ("walk",)

		o1 = action_history[1]
		o2 = action_history[3]
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

		util_vec = compute_utility(is_deal, pool, v1, v2, action_history[-2][0], o1_pay, o2_pay, num_prev_rounds)
		return util_vec * input_reach_prob, POLICY_SPACE1, POLICY_SPACE2
	else:
		pay = np.zeros(N)
		player_num = 1
		v = v1
		PS = POLICY_SPACE1.copy()
		empir_infoset_id = get_empirical_infoset_id_given_histories(action_history, pool, POLICY_SPACE1, POLICY_SPACE2)

		if len(action_history) % 2 == 1:
			player_num = 2
			v = v2
			PS = POLICY_SPACE2.copy()

		assert empir_infoset_id[0] == player_num

		infoset_strat = strategy_profile.get(empir_infoset_id)
		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))
		game_start = len(action_history) <= 4

		if infoset_strat is not None:
			for empir_action in infoset_strat.keys():
				#print("empir_action ", empir_action)
				policy_str = empir_action[0]
				if infoset_strat.get(empir_action) > 0.0:
					offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
					next_node = action_history + ((offer, empir_action[1]),)
					next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)

					new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

		else:
			signal = random.choice([True, False])
			policy_str = None
			if len(action_history) <= 5:
				policy_str = "pi_0"
			else:
				last_policy_str = empir_infoset_id[1][-2][0]
				#print("hey")
				if last_policy_str[0] is None:
					last_policy_str = "pi_0"

				policy_str = last_policy_str

			empir_action = (policy_str, signal)
			#print("empir_action ", empir_action)
			offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
			#print("offer ", offer)
			a = (offer, signal)
			next_node = action_history + (a,)
			next_reach_prob = input_reach_prob

			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

	return pay, POLICY_SPACE1, POLICY_SPACE2

def compute_regret(meta_strategy, eval_string, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, file_ID, hp_set1, hp_set2, 
	POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
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
	meta_strategy_pay, POLICY_SPACE1, POLICY_SPACE2 = compute_true_empirical_strategy_pay(meta_strategy, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)

	print("true ms pay ", meta_strategy_pay)
	print("computing regret")
	print(eval_string)
	regrets = []

	# BR1, BR2, POLICY_SPACE1, POLICY_SPACE2, br1_payoffs_over_time, br2_payoffs_over_time = compute_best_response(meta_strategy, eval_string, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	# 	o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, 1000, True)
	BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(meta_strategy, eval_string, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
		o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, 1000, True)

	print("we haz BR now")

	for j in range(2):
		print("j ", j)

		action_pay = None
		if j == 0:
			action_pay, POLICY_SPACE1, POLICY_SPACE2 = compute_true_pay(meta_strategy, BR1_weights, BR2_weights, 1, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			
		else:
			action_pay, POLICY_SPACE1, POLICY_SPACE2 = compute_true_pay(meta_strategy, BR1_weights, BR2_weights, 2, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)

		print("action_pay ", action_pay)
		print("ms pay ", meta_strategy_pay)
		regrets.append(max(action_pay[j] - meta_strategy_pay[j], 0.0))

	print("max regrets ", max(regrets))
	return max(regrets)

def construct_initial_policy(game_param_map, hp_set):
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

	#print("offer_space ", offer_space)

	initial_policy = {}

	POLICY_SPACE1 = {}
	POLICY_SPACE2 = {}

	player1_infosets = [(1, ("H",)), (1, ("L",))]

	state_len = get_state_length(pool)
	print("state_len ", state_len)
	action_space = list(it.product(offer_space, [True, False]))

	action_space1_start = list(it.product(offer_space1, [True, False]))

	default_start_model1 = tf.keras.Sequential()
	default_start_model1.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_start_model1.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))

	default_start_model1.add(tf.keras.layers.Dense(len(action_space1_start)))
	default_start_model1.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))

	W1_start = default_start_model1.get_weights()

	#print("W1_start ", len(W1_start))

	default_model1 = tf.keras.Sequential()

	default_model1.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_model1.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))

	default_model1.add(tf.keras.layers.Dense(len(action_space)))
	default_model1.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))

	W1_default = default_model1.get_weights()
	#print("W1_default ", len(W1_default[0]))

	default_model2 = tf.keras.Sequential()

	default_model2.add(tf.keras.layers.Dense(hp_set["model_width"], input_shape=(state_len,), activation="relu"))
	default_model2.add(tf.keras.layers.Dense(hp_set["model_width"], activation="relu"))

	default_model2.add(tf.keras.layers.Dense(len(action_space)))
	default_model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=hp_set["learning_rate"]))

	W2_default = default_model2.get_weights()

	reveal1L, reveal1H, reveal1, reveal2 = np.random.choice([True, False], 4)


	# Do I include the T/F as part of the action space in the neural networks (and therefore the true game)?
	# TBD
	initial_policy[(1, ("H",))] = {("pi_0", reveal1H): 1.0}
	#initial_policy[(1, ("H",))] = {("pi_0", reveal1H): 0.8, ("pi_0", not reveal1H): 0.2}
	initial_policy[(1, ("L",))] = {("pi_0", reveal1L): 1.0}
	#initial_policy[(1, ("L",))] = {("pi_0", reveal1L): 1.0 / 3, ("pi_0", not reveal1L): 2.0 / 3}

	if reveal1H:

		initial_policy[(2, ("H", "H", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("H", "L", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}

	else:
		initial_policy[(2, ("H", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("L", ("pi_0", reveal1H)))] = {("pi_0", reveal2): 1.0}

	if reveal1L:
		initial_policy[(2, ("L", "H", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("L", "L", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}

	else:
		initial_policy[(2, ("H", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}
		initial_policy[(2, ("L", ("pi_0", reveal1L)))] = {("pi_0", reveal2): 1.0}

	POLICY_SPACE1["pi_0"] = W1_start
	POLICY_SPACE2["pi_0"] = W2_default

	print("initial_policy ")
	for x in initial_policy:
		print(x, initial_policy[x])
	
	return initial_policy, ("pi_0", bool(reveal1)), ("pi_0", bool(reveal2)), POLICY_SPACE1, POLICY_SPACE2


def compute_empirical_pay_given_infoset(meta_strategy, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	return recursive_empirical_pay_infoset_helper(meta_strategy, [], [], 1.0, 1.0, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
		o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)


def recursive_empirical_pay_infoset_helper(meta_strategy, true_history, empir_history, input_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	turns_taken = len(true_history) - 4
	num_prev_rounds = turns_taken // 2
	infoset_freq = None

	# print("true_history ", true_history)
	# print("input infoset_id ", infoset_id)
	# print("meta_strategy ", meta_strategy)
	# print("infoset_reach_prob ", infoset_reach_prob)
	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(true_history) == 0:
		pay = 0.0
		infoset_freq = 0.0

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			infoset_freq = new_infoset_freq
	
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(true_history) == 1:
		pay = 0.0
		infoset_freq = 0.0
		br_player = infoset_id[0]
		if br_player == 1:
			o1 = infoset_id[1][0]
			#print("o1 ", o1)
			#prob = 1.0
			next_node = true_history + (o1,)
			next_empir_history = (o1,)
			next_reach_prob = input_reach_prob
			next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist1.get(o1)
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			#print("infoset_freq ", new_infoset_freq)
			#print(hhh)
			infoset_freq = new_infoset_freq

		else:
			# handle player 2 later
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			#print(signal_count)

			# this is player 2's infoset
			if signal_count == 2:
				# o1 = empir_history[0]
				o1 = infoset_id[1][0]
				#print("o1 revealed", o1)
				next_node = true_history + (o1,)
				next_empir_history = (o1,)
				next_reach_prob = input_reach_prob
				next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist1.get(o1)
				#print("next_infoset_reach_prob ", next_infoset_reach_prob)
				new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				#print("infoset_freq ", infoset_freq)
				infoset_freq = new_infoset_freq
				#print("upd infoset_freq ", infoset_freq)
				#print(hhh)
			else:

				for o1 in OUTSIDE_OFFERS:
					prob = outside_offer_dist1.get(o1)
					#print("o1 ", o1)
					#print("prob ", prob)
					next_node = true_history + (o1,)
					next_empir_history = (o1,)
					next_reach_prob = input_reach_prob * prob
					next_infoset_reach_prob = infoset_reach_prob * prob
					#print("next_infoset_reach_prob ", next_infoset_reach_prob)
					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					#print("infoset_freq ", infoset_freq)
					infoset_freq = infoset_freq + new_infoset_freq
					#print("updated infoset freq ", infoset_freq)
					#print(hhh)

	# third chance node has been reached -- valuation for player 2 in true game
	elif len(true_history) == 2:
		pay = 0.0
		infoset_freq = 0.0
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = true_history + (v2,)
			next_reach_prob = input_reach_prob * prob

			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			#print("infoset_freq ", infoset_freq)
			infoset_freq = new_infoset_freq
			#print("infoset_freq? ", infoset_freq)
			#print(hhh)

	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(true_history) == 3:
		pay = 0.0
		infoset_freq = 0.0
		br_player = infoset_id[0]
		if br_player == 1:

			# this infoset belongs to player 1
			# offer_reveal = check_empirical_outside_offer_reveal(infoset_id[1], 2)
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			#print(signal_count)
			if signal_count == 2:
				o2 = infoset_id[1][1]
				#print("o2 ", o2)
				#print(infoset_id[1][1])
				# print(yyyyy)
				next_node = true_history + (o2,)
				next_empir_history = empir_history + (o2,)
				next_reach_prob = input_reach_prob
				next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist2.get(o2)
				#print("next_infoset_reach_prob ", next_infoset_reach_prob)
				#print(yyyyy)
				new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				#print("infoset_freq ", infoset_freq)
				#print(new_infoset_freq)
				infoset_freq = new_infoset_freq
				#print(hhh)
			else:
				for o2 in OUTSIDE_OFFERS:
					#print("o2! ", o2)
					prob = outside_offer_dist2.get(o2)
					next_node = true_history + (o2,)
					next_empir_history = empir_history + (o2,)
					next_reach_prob = input_reach_prob * prob
					next_infoset_reach_prob = infoset_reach_prob * prob
					#print("prob ", prob)
					#print("next_infoset_reach_prob ", next_infoset_reach_prob)

					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					#print("infoset_freq! ", infoset_freq)
					#print(new_infoset_freq)
					infoset_freq = infoset_freq + new_infoset_freq
					#infoset_freq = new_infoset_freq
				#print(hhh)

		else:
			#print("empir_history ", empir_history)
			#print(infoset_id)
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			#print(signal_count)
			# offer_reveal = check_empirical_outside_offer_reveal(infoset_id[1], 1)
			# print(offer_reveal)
			o2 = None
			if signal_count == 2:
				o2 = infoset_id[1][1]
			else:
				o2 = infoset_id[1][0]
			#print("o2 ", o2)
			# print(stahp)

			# prob = outside_offer_dist2.get(o2)
			next_node = true_history + (o2,)
			next_empir_history = empir_history + (o2,)
			next_reach_prob = input_reach_prob
			next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist2.get(o2)
			#print("next_infoset_reach_prob ", next_infoset_reach_prob)
			
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			#print("infoset_freq ", infoset_freq)
			infoset_freq = new_infoset_freq
			#print("hey itsa infoset_freq ", infoset_freq)
			#print(hyeyeye)

	elif true_history[-1][0] in [("walk",), ("deal",)] or turns_taken >= NUM_PLAYER_TURNS * 2:
		br_player = infoset_id[0]
		is_deal = true_history[-1][0]
		if is_deal not in [("walk",), ("deal",)]:
			is_deal = ("walk",)

		v1 = true_history[0]
		o1 = true_history[1]
		v2 = true_history[2]
		o2 = true_history[3]
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

		util_vec = compute_utility(is_deal, pool, v1, v2, true_history[-2][0], o1_pay, o2_pay, num_prev_rounds)
		return util_vec[br_player - 1] * input_reach_prob, infoset_reach_prob

	else:
		pay = 0.0
		infoset_freq = 0.0
		player_num = 1
		br_player = infoset_id[0]

		v1 = true_history[0]
		v = v1
		PS = POLICY_SPACE1.copy()
		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))

		if len(true_history) % 2 == 1:
			player_num = 2
			v2 = true_history[2]
			v = v2
			PS = POLICY_SPACE2.copy()

		#print("player_num ", player_num)
		#print("empir_history ", empir_history)

		if player_num != br_player:
			#print("other")
			#print("empir_history ", empir_history)
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, br_player)
			#empir_infoset_id = get_empirical_infoset_id_given_histories(true_history, pool, POLICY_SPACE1, POLICY_SPACE2)
			empir_infoset_id = None
			if offer_reveal:
				empir_infoset_id = (player_num, empir_history)

			else:
				#print("noreveal")
				if br_player == 1:
					empir_infoset_id = (player_num, empir_history[1:])
				else:
					empir_infoset_id = (player_num, (empir_history[0],) + empir_history[2:])

			#print(empir_infoset_id)
			assert empir_infoset_id[0] == player_num
			# print(nahhhh)
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			#print(empir_actions)
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			#print(input_empir_actions)
			if len(empir_actions) < len(input_empir_actions):
				# choose other player's actions w/ prob 1 so that they lead to the input br_player's infoset
				infoset_freq = 0.0
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				#print("empir_action ", empir_action)
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				#print("next_node ", next_node)
				next_empir_history = empir_history + (empir_action,)
				#print("next_empir_history ", next_empir_history)
				#else:
				infoset_strat = meta_strategy.get(empir_infoset_id)
				if infoset_strat is not None:
					#print("infoset_strat ", infoset_strat)
					prob = infoset_strat.get(empir_action, 0.0)
					if prob > 0.0:
						#print("prob ", prob)
						next_infoset_reach_prob = prob * infoset_reach_prob
						#print("next_infoset_reach_prob ", next_infoset_reach_prob)
						new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
							outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
						pay = pay + new_pay
						#print("infoset_freq ", infoset_freq)
						infoset_freq = infoset_freq + new_infoset_freq
						#infoset_freq = new_infoset_freq
						#print("infoset_freqoo ", infoset_freq)
						#print(hddhh)

			else:
				# follow meta_strategy after input br_player's infoset
				infoset_strat = meta_strategy.get(empir_infoset_id)
				game_start = len(true_history) <= 4

				if infoset_strat is not None:
					infoset_freq = 0.0
					for empir_action in infoset_strat.keys():
						#print("empir_action ", empir_action)
						policy_str = empir_action[0]
						if infoset_strat.get(empir_action, 0.0) > 0.0:
							offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
							next_node = true_history + ((offer, empir_action[1]),)
							#print("next_node ", next_node)
							next_empir_history = empir_history + (empir_action,)
							#print(elliot)
							next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)

							new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
								outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
							pay = pay + new_pay
							# print("empir_history ", empir_history)
							# print("empir_infoset_id ", empir_infoset_id)
							# print(infoset_strat)
							# print("infoset_freq ", infoset_freq)
							infoset_freq = new_infoset_freq
							#print("infoset_freqyy ", infoset_freq)
							#print(jfjkl)

				else:
					infoset_freq = 0.0
					signal = random.choice([True, False])
					
					if len(true_history) <= 5:
						policy_str = "pi_0"
					else:
						last_policy_str = empir_infoset_id[1][-2][0]
						#print("hey")
						if last_policy_str[0] is None:
							last_policy_str = "pi_0"

						policy_str = last_policy_str

					empir_action = (policy_str, signal)
					offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
					a = (offer, signal)
					next_node = true_history + (a,)
					next_empir_history = empir_history + (empir_action,)
					next_reach_prob = input_reach_prob

					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					#print("infoset_freq! ", infoset_freq)
					infoset_freq = new_infoset_freq
					#print("infoset_freq ", infoset_freq)
					#print(yfyfyfyf)

		else:
			
			other_player = player_num % 2 + 1
			#print("other_player ", other_player)
			#print("empir_history ", empir_history)
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, other_player)
			#print("offer_reveal ", offer_reveal)
			empir_infoset_id = None
			if offer_reveal:
				empir_infoset_id = (br_player, empir_history)
			else:
				#print("noreveal")
				if other_player == 1:
					empir_infoset_id = (br_player, empir_history[1:])
				else:
					empir_infoset_id = (br_player, (empir_history[0],) + empir_history[2:])
			
			#print("empir_infoset_id ", empir_infoset_id)
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			#print(empir_actions)
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			#print(input_empir_actions)
			#print(infoset_id)
			#print("true_history ", true_history)
			
			# choose br_player's actions so they lead to given infoset
			if len(empir_actions) < len(input_empir_actions):
				infoset_freq = 0.0
				#print("hoo")
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				#print("empir_action ", empir_action)
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				#print("next_node ", next_node)
				next_empir_history = empir_history + (empir_action,)
				#print("next_empir_history ", next_empir_history)
				infoset_strat = meta_strategy.get(empir_infoset_id)
				#print("infoset_strat ", infoset_strat)
				prob = infoset_strat.get(empir_action, 0.0)
				#print("prob ", prob)
				if prob > 0.0:
					next_infoset_reach_prob = infoset_reach_prob * prob
					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					#print("infoset_freqqqq ", infoset_freq)
					infoset_freq = infoset_freq + new_infoset_freq
					#print("nn infoset_freq ", infoset_freq)
				#print("prob was zero?")

				#print(nddah)


			else:
				infoset_strat = meta_strategy.get(empir_infoset_id)
				game_start = len(true_history) <= 4

				if infoset_strat is not None:
					infoset_freq = 0.0
					for empir_action in infoset_strat.keys():
						#print("empir_action ", empir_action)
						policy_str = empir_action[0]
						if infoset_strat.get(empir_action, 0.0) > 0.0:
							offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
							next_node = true_history + ((offer, empir_action[1]),)
							#print("next_node ", next_node)
							next_empir_history = empir_history + (empir_action,)
							#print(elliot)
							next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)

							new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
								outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
							pay = pay + new_pay
							#print("infoset_freq ", infoset_freq)
							#print("new_infoset_freq heyo ", new_infoset_freq)
							#print(nah)
							# infoset_freq = infoset_freq + new_infoset_freq
							infoset_freq = new_infoset_freq

				else:
					infoset_freq = 0.0
					signal = random.choice([True, False])
					
					if len(true_history) <= 5:
						policy_str = "pi_0"
					else:
						last_policy_str = empir_infoset_id[1][-2][0]
						#print("hey")
						if last_policy_str[0] is None:
							last_policy_str = "pi_0"

						policy_str = last_policy_str

					empir_action = (policy_str, signal)
					offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
					a = (offer, signal)
					next_node = true_history + (a,)
					next_empir_history =empir_history + (empir_action,)
					next_reach_prob = input_reach_prob

					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					#print("infoset_freq ", infoset_freq)
					#print("new_infoset_freq ", new_infoset_freq)
					# infoset_freq = infoset_freq + new_infoset_freq
					infoset_freq = new_infoset_freq
					#print(hhhhh)


	return pay, infoset_freq

def compute_infoset_gain_of_best_response(meta_strategy, infoset_id, br_player, BR_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	empirical_pay, infoset_freq = compute_empirical_pay_given_infoset(meta_strategy, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)

	br_pay = recursive_infoset_gain_helper(meta_strategy, [], [], 1.0, infoset_id, br_player, BR_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)

	gain = br_pay - empirical_pay
	print("gain ", gain, "\n\n")

	return gain, infoset_freq

def recursive_infoset_gain_helper(meta_strategy, true_history, empir_history, input_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	turns_taken = len(true_history) - 4
	num_prev_rounds = turns_taken // 2

	#print("true_history ", true_history)
	#print("input infoset_id ", infoset_id)
	#print("meta_strategy ", meta_strategy)
	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(true_history) == 0:
		pay = 0.0

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(true_history) == 1:
		pay = 0.0
		if br_player == 1:
			o1 = infoset_id[1][0]
			#print("o1 ", o1)
			#prob = 1.0
			next_node = true_history + (o1,)
			next_empir_history = (o1,)
			next_reach_prob = input_reach_prob
			new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

		else:
			# handle player 2 later
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			#print(signal_count)

			# this is player 2's infoset
			if signal_count == 2:
				# o1 = empir_history[0]
				o1 = infoset_id[1][0]
				#print("o1 revealed", o1)
				next_node = true_history + (o1,)
				next_empir_history = (o1,)
				next_reach_prob = input_reach_prob
				new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
			else:

				for o1 in OUTSIDE_OFFERS:
					prob = outside_offer_dist1.get(o1)
					next_node = true_history + (o1,)
					next_empir_history = (o1,)
					next_reach_prob = input_reach_prob * prob
					new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

	# third chance node has been reached -- valuation for player 2 in true game
	elif len(true_history) == 2:
		pay = 0.0
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = true_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay	

	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(true_history) == 3:
		pay = 0.0
		if br_player == 1:

			# this infoset belongs to player 1
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			if signal_count == 2:
				o2 = infoset_id[1][1]
				#print("o2 ", o2)
				#print(infoset_id[1][1])
				# print(yyyyy)
				next_node = true_history + (o2,)
				next_empir_history = empir_history + (o2,)
				next_reach_prob = input_reach_prob
				new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
			else:
				for o2 in OUTSIDE_OFFERS:
					prob = outside_offer_dist2.get(o2)
					next_node = true_history + (o2,)
					next_empir_history = empir_history + (o2,)
					next_reach_prob = input_reach_prob * prob
					new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

		else:
			# print("empir_history ", empir_history)
			# print(infoset_id)
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			# print(signal_count)
			o2 = None
			if signal_count == 2:
				o2 = infoset_id[1][1]
			else:
				o2 = infoset_id[1][0]
			#print("o2 ", o2)

			# prob = outside_offer_dist2.get(o2)
			next_node = true_history + (o2,)
			next_empir_history = empir_history + (o2,)
			next_reach_prob = input_reach_prob
			new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

	elif true_history[-1][0] in [("walk",), ("deal",)] or turns_taken >= NUM_PLAYER_TURNS * 2:

		is_deal = true_history[-1][0]
		if is_deal not in [("walk",), ("deal",)]:
			is_deal = ("walk",)

		v1 = true_history[0]
		o1 = true_history[1]
		v2 = true_history[2]
		o2 = true_history[3]
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

		util_vec = compute_utility(is_deal, pool, v1, v2, true_history[-2][0], o1_pay, o2_pay, num_prev_rounds)
		return util_vec[br_player - 1] * input_reach_prob

	else:
		pay = 0.0
		player_num = 1

		v1 = true_history[0]
		v = v1
		PS = POLICY_SPACE1.copy()
		offer_space = generate_offer_space(pool)
		action_space = list(it.product(offer_space, [True, False]))

		if len(true_history) % 2 == 1:
			player_num = 2
			v2 = true_history[2]
			v = v2
			PS = POLICY_SPACE2.copy()

		# print("player_num ", player_num)
		# print("empir_history ", empir_history)

		if player_num != br_player:
			# print("other")
			# print("empir_history ", empir_history)
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, br_player)
			#empir_infoset_id = get_empirical_infoset_id_given_histories(true_history, pool, POLICY_SPACE1, POLICY_SPACE2)
			empir_infoset_id = None
			if offer_reveal:
				empir_infoset_id = (player_num, empir_history)

			else:
				#print("noreveal")
				if br_player == 1:
					empir_infoset_id = (player_num, empir_history[1:])
				else:
					empir_infoset_id = (player_num, (empir_history[0],) + empir_history[2:])

			#print(empir_infoset_id)
			assert empir_infoset_id[0] == player_num
			# print(nahhhh)
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			#print(empir_actions)
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			#print(input_empir_actions)
			if len(empir_actions) <= len(input_empir_actions):
				# choose other player's actions w/ prob 1 so that they lead to the input br_player's infoset
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				#print("empir_action ", empir_action)
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				#print("next_node ", next_node)
				next_empir_history = empir_history + (empir_action,)
				#print("next_empir_history ", next_empir_history)
				new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
			else:
				# follow meta_strategy after input br_player's infoset
				infoset_strat = meta_strategy.get(empir_infoset_id)
				game_start = len(true_history) <= 4

				if infoset_strat is not None:
					for empir_action in infoset_strat.keys():
						#print("empir_action ", empir_action)
						policy_str = empir_action[0]
						if infoset_strat.get(empir_action) > 0.0:
							offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
							next_node = true_history + ((offer, empir_action[1]),)
							#print("next_node ", next_node)
							next_empir_history = empir_history + (empir_action,)
							#print(elliot)
							next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)

							new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
								outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
							pay = pay + new_pay

				else:
					signal = random.choice([True, False])
					
					if len(true_history) <= 5:
						policy_str = "pi_0"
					else:
						last_policy_str = empir_infoset_id[1][-2][0]
						#print("hey")
						if last_policy_str[0] is None:
							last_policy_str = "pi_0"

						policy_str = last_policy_str

					empir_action = (policy_str, signal)
					offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
					a = (offer, signal)
					next_node = true_history + (a,)
					next_empir_history =empir_history + (empir_action,)
					next_reach_prob = input_reach_prob

					new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

		else:
			other_player = player_num % 2 + 1
			#print("empir_history ", empir_history)
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, other_player)
			#print("offer_reveal ", offer_reveal)
			empir_infoset_id = None
			if offer_reveal:
				empir_infoset_id = (br_player, empir_history)
			else:
				#print("noreveal")
				if other_player == 1:
					empir_infoset_id = (br_player, empir_history[1:])
				else:
					empir_infoset_id = (br_player, (empir_history[0],) + empir_history[2:])
			
			#print("empir_infoset_id ", empir_infoset_id)
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			#print(empir_actions)
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			# print(input_empir_actions)
			
			# choose br_player's actions so they lead to given infoset
			if len(empir_actions) < len(input_empir_actions):
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				#print("empir_action ", empir_action)
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				next_empir_history = empir_history + (empir_action,)
				new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist,
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay

			else:
				state = convert_into_state(true_history, pool)

				best_action = get_best_action(state, BR_weights, action_space, len(true_history) <= 4)

				next_node = true_history + (best_action,)
				next_reach_prob = input_reach_prob
				# print("next_node ", next_node)
				# for k in POLICY_SPACE1:
				# 	print(k)
				policy_str = "pi_" + str(len(PS) - 1)
				#print("ps ", policy_str)
				next_empir_history = empir_history + ((policy_str, best_action[1]),)
				#print("next_empir_history ", next_empir_history)
				new_pay = recursive_infoset_gain_helper(meta_strategy, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				
	return pay

def te_egta(game_param_map, T, trial_index):
	'''
	@arg (map) initial_sigma: Initial metastrategy based on empirical strategy
		space
	@arg (map) game_param_map: map of game parameters for given file ID (player valuation distribution, item pool,
		outside offer distributions)
	@arg (int) T: Number of iterations for a single run of CFR (whether solving a game for NE or a subgame as part
		of solving a game for SPE)

	Runs a single play of TE-EGTA on large abstract game, expanding strategy space, simulating
	each new strategy and constructing the empirical game model, which is then solved for an
	approximate NE using counter-factual regret minimization (CFR)
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

	empir_root = Node(0, (0, 1), [], [], N)
	X = {}

	initial_sigma, default_policy1, default_policy2, POLICY_SPACE1, POLICY_SPACE2 = construct_initial_policy(game_param_map, hp_set1)

	# Initialize the empirical strategy space based on initial_sigma
	empir_strat_space = {}
	for i in initial_sigma.keys():
		empir_strat_space[i] = [list(initial_sigma[i].keys())[0]]

	prefix = 'BIG_DoND_TE-PSRO_BR' + str(NUM_EMPIR_BR) + '_' + str(trial_index) + '_' + file_ID
	print(prefix)
	file_name = prefix + '_empirical_game.mmdpickle'
	with open(file_name, 'w') as fp:
		pass
	mgame = mmp.mmapdict(file_name)
	# '''
	empirical_game = ExtensiveForm([[], []], empir_root, [], {}, NUM_PLAYER_TURNS)
	mgame['game'] = empirical_game

	regret_over_time = []
	payoff_estimation_error_over_time = []
	# br1_payoffs_over_time = []
	# br2_payoffs_over_time = []
	# runtimes = []

	payoffs = {}
	observations = {}

	meta_strategy = initial_sigma.copy()
	empirical_game_size_over_time = []
	solution_over_time = []
	
	regret = compute_regret(meta_strategy, "", pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, prefix, 
		hp_set1, hp_set2, POLICY_SPACE1.copy(), POLICY_SPACE2.copy(), default_policy1, default_policy2)
	regret_over_time.append(regret)
	#regret_over_time = [1.0]
	print("regret_over_time so far ", regret_over_time)
	
	BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(meta_strategy, "", prefix, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
		o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, NUM_EMPIR_BR, False)
	
	player1_empir_infostates = [infoset_id for infoset_id in meta_strategy if infoset_id[0] == 1]
	num_br_samples = min(NUM_EMPIR_BR, len(player1_empir_infostates))
	policy_str = "pi_" + str(len(POLICY_SPACE1))
	POLICY_SPACE1[policy_str] = BR1_weights

	infoset_gains1 = []

	for infoset_id in player1_empir_infostates:
		#print("infoset_id1 ", infoset_id)
		infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(meta_strategy, infoset_id, 1, BR1_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
			POLICY_SPACE1, POLICY_SPACE2)
		# print("infoset_gain1 ", infoset_gain)
		# print(infoset_freq)
		infoset_gains1.append(infoset_gain * infoset_freq)

	for i in range(len(infoset_gains1)):
		print(player1_empir_infostates[i])
		print(infoset_gains1[i])
	print("\n")
	softmax_gains1 = softmax(infoset_gains1)
	#print("softmax_gains1 ", softmax_gains1)

	x = np.arange(len(player1_empir_infostates))
	#print("x ", x)
	infoset_inds_1 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax_gains1)
	#print("infoset_inds_1 ", infoset_inds_1)
	player1_empir_M = [player1_empir_infostates[i] for i in infoset_inds_1]
	print("player1_empir_M ", player1_empir_M)
	BR1 = convert_into_best_response_policy(player1_empir_M, BR1_weights, policy_str)
	print("BR1 ", BR1)

	infoset_gains2 = []
	player2_empir_infostates = [infoset_id for infoset_id in meta_strategy if infoset_id[0] == 2]
	num_br_samples = min(NUM_EMPIR_BR, len(player2_empir_infostates))
	policy_str = "pi_" + str(len(POLICY_SPACE2))
	POLICY_SPACE2[policy_str] = BR2_weights

	for infoset_id in player2_empir_infostates:
		#print("infoset_id2 ", infoset_id)
		infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(meta_strategy, infoset_id, 2, BR2_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
			POLICY_SPACE1, POLICY_SPACE2)
		# print("infoset_gain2 ", infoset_gain)
		# print(infoset_freq)
		infoset_gains2.append(infoset_gain * infoset_freq)

	for i in range(len(infoset_gains2)):
		print(player2_empir_infostates[i])
		print(infoset_gains2[i])
	print("\n")
	softmax_gains2 = softmax(infoset_gains2)
	#print("softmax_gains2 ", softmax_gains2)
	x = np.arange(len(player2_empir_infostates))
	#print("x ", x)
	infoset_inds_2 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax_gains2)
	#print("infoset_inds_2 ", infoset_inds_2)
	player2_empir_M = [player2_empir_infostates[i] for i in infoset_inds_2]
	print("player2_empir_M ", player2_empir_M)
	BR2 = convert_into_best_response_policy(player2_empir_M, BR2_weights, policy_str)
	print("BR2 ", BR2)
	

	BR = {}
	BR.update(BR1)
	BR.update(BR2)

	empirical_game_size_over_time.append([len(empir_strat_space), total_size(empirical_game, HANDLERS)])
	print("empirical_game_size_over_time so far ", empirical_game_size_over_time)

	#regret_over_time.append(7)

	# save policy maps to disk
	np.save(prefix + "_policy_map1.npy", POLICY_SPACE1)
	np.save(prefix + "_policy_map2.npy", POLICY_SPACE2)

	while len(empirical_game_size_over_time) < NUM_PSRO_ITER and (regret_over_time[-1] > REGRET_THRESH):
		old_empir_strat_space = empir_strat_space.copy()

		# Load policy maps from disk
		POLICY_SPACE1 = np.load(prefix + "_policy_map1.npy", allow_pickle=True).item()
		POLICY_SPACE2 = np.load(prefix + "_policy_map2.npy", allow_pickle=True).item()

		print("simulating true game and updating empirical game w/ new simulation data")
		total_NF_sample_budget = get_total_nf_budget(SAMPLING_BUDGET, len(empirical_game_size_over_time))
		print("total_NF_sample_budget ", total_NF_sample_budget)
		
		# Simulate initial policy as well at start of TE-PSRO
		if len(empirical_game_size_over_time) == 1:
			payoffs, observations = simulate(val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr,
				pool, empir_strat_space, {}, SAMPLING_BUDGET, STD_DEV, payoffs, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2)

			empirical_game = mgame['game']
			empirical_game.update_game_with_simulation_output(observations, payoffs)
			mgame['game'] = empirical_game
			del observations


		payoffs, new_observations = simulate(val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, pool, empir_strat_space, BR, total_NF_sample_budget, STD_DEV, payoffs, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2)
		
		#print("new_observations ", new_observations)
		empirical_game = mgame['game']
		# empirical_game.update_game_with_simulation_output(observations, payoffs)
		empirical_game.update_game_with_simulation_output(new_observations, payoffs)
		mgame['game'] = empirical_game
		del new_observations
		
		for i in range(2):
			for infoset in empirical_game.infosets[i]:
				infoset_id = infoset.infoset_id
				print("k ", infoset_id)
				print(empir_strat_space.get(infoset_id))
				if infoset_id not in old_empir_strat_space.keys():
					empir_strat_space[infoset_id] = infoset.action_space[:]

				else:
					empir_strat_space[infoset_id] = infoset.action_space[:]

				print("updated ", empir_strat_space.get(infoset_id))


		empirical_game_size_over_time.append([len(empir_strat_space), total_size(empirical_game, HANDLERS)])
		print("empirical_game_size_over_time so far ", empirical_game_size_over_time)

		num_combs = 0
		avg_err = 0
		#print("ESS ", empir_strat_space)

		# s_comb_list = [dict(zip(empir_strat_space, p)) for p in it.product(*empir_strat_space.values())]
		s_comb_list = [dict(zip(empir_strat_space, p)) for p in it.product(*empir_strat_space.values())]

		for s_comb in s_comb_list:
			print("s_comb ", s_comb)

			sigma = {}
			for k in s_comb:
				sigma[k] = {s_comb[k]: 1.0}

			# sigma = {}
			# for k in range(len(empir_strat_space.keys())):
			# 	key_list = list(empir_strat_space.keys())
			# 	infoset_id = key_list[k]
			# 	sigma[infoset_id] = {s_comb[k]: 1.0}
			
			# Compute and record payoff estimation error
			#true_pay, strategy = compute_true_pay(sigma, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			true_pay, POLICY_SPACE1, POLICY_SPACE2 = compute_true_empirical_strategy_pay(sigma, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			print("true_pay ", true_pay)

			est_pay = empirical_game.compute_pay(sigma)
			print("est_pay ", est_pay)

			est_err = max(abs(est_pay - true_pay))
			avg_err += est_err
			num_combs += 1

			if num_combs == 500:
				break
		
		print("num_combs ", num_combs)
		avg_err /= num_combs

		print("error_over_time ", avg_err)
		payoff_estimation_error_over_time.append(avg_err)

		#print(off)

		print("computing new metastrategy")
		empirical_game_copy1 = copy.deepcopy(empirical_game)
		meta_strategy = empirical_game.cfr(T)

		#print(stahp)
		print("computing regret with new metastrategy")
		regret = compute_regret(meta_strategy, "", pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
			prefix, hp_set1, hp_set2, POLICY_SPACE1.copy(), POLICY_SPACE2.copy(), default_policy1, default_policy2)
		regret_over_time.append(regret)

		BR1_weights, BR2_weights, POLICY_SPACE1, POLICY_SPACE2 = compute_best_response(meta_strategy, "", prefix, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
			o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, NUM_EMPIR_BR, False)

		player1_empir_infostates = [infoset_id for infoset_id in meta_strategy if infoset_id[0] == 1]
		num_br_samples = min(NUM_EMPIR_BR, len(player1_empir_infostates))
		policy_str = "pi_" + str(len(POLICY_SPACE1))
		POLICY_SPACE1[policy_str] = BR1_weights

		infoset_gains1 = []

		for infoset_id in player1_empir_infostates:
			#print("infoset_id1 ", infoset_id)
			infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(meta_strategy, infoset_id, 1, BR1_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
				POLICY_SPACE1, POLICY_SPACE2)
			# print("infoset_gain1 ", infoset_gain)
			# print(infoset_freq)
			infoset_gains1.append(infoset_gain * infoset_freq)

		for i in range(len(infoset_gains1)):
			print(player1_empir_infostates[i])
			print(infoset_gains1[i])
		print("\n")
		softmax_gains1 = softmax(infoset_gains1)
		# print("softmax_gains1 ", softmax_gains1)

		x = np.arange(len(player1_empir_infostates))
		#print("x ", x)
		infoset_inds_1 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax_gains1)
		#print("infoset_inds_1 ", infoset_inds_1)
		player1_empir_M = [player1_empir_infostates[i] for i in infoset_inds_1]
		print("player1_empir_M ", player1_empir_M)
		BR1 = convert_into_best_response_policy(player1_empir_M, BR1_weights, policy_str)
		print("BR1 ", BR1)

		infoset_gains2 = []
		player2_empir_infostates = [infoset_id for infoset_id in meta_strategy if infoset_id[0] == 2]
		num_br_samples = min(NUM_EMPIR_BR, len(player2_empir_infostates))
		policy_str = "pi_" + str(len(POLICY_SPACE2))
		POLICY_SPACE2[policy_str] = BR2_weights

		for infoset_id in player2_empir_infostates:
			#print("infoset_id2 ", infoset_id)
			infoset_gain, infoset_freq = compute_infoset_gain_of_best_response(meta_strategy, infoset_id, 2, BR2_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
				POLICY_SPACE1, POLICY_SPACE2)
			# print("infoset_gain2 ", infoset_gain)
			# print(infoset_freq)
			infoset_gains2.append(infoset_gain * infoset_freq)

		for i in range(len(infoset_gains2)):
			print(player2_empir_infostates[i])
			print(infoset_gains2[i])
		print("\n")
		softmax_gains2 = softmax(infoset_gains2)
		#print("softmax_gains2 ", softmax_gains2)
		x = np.arange(len(player2_empir_infostates))
		#print("x ", x)
		infoset_inds_2 = np.random.choice(x, size=num_br_samples, replace=False, p=softmax_gains2)
		#print("infoset_inds_2 ", infoset_inds_2)
		player2_empir_M = [player2_empir_infostates[i] for i in infoset_inds_2]
		print("player2_empir_M ", player2_empir_M)
		BR2 = convert_into_best_response_policy(player2_empir_M, BR2_weights, policy_str)
		print("BR2 ", BR2)
		#print(hiissss)

		BR = {}
		BR.update(BR1)
		BR.update(BR2)

		# save policy maps to disk
		np.save(prefix + "_policy_map1.npy", POLICY_SPACE1)
		np.save(prefix + "_policy_map2.npy", POLICY_SPACE2)

		np.savez_compressed(prefix, regret_over_time, empirical_game_size_over_time, payoff_estimation_error_over_time)
		#print(stahp)

	del mgame['game']
	mgame.vacuum()


file_ID_index = int(sys.argv[1]) // 5
print(file_ID_index)

trial_index = int(sys.argv[1]) % 5
print(trial_index)

br_index = int(sys.argv[2])
print(br_index)


file_ID = file_ID_list[file_ID_index]
print("file_ID ", file_ID)

NUM_EMPIR_BR = emp_br_list[br_index]
print("NUM_EMPIR_BR ", NUM_EMPIR_BR)

game_params = retrieve_game(file_ID_list[file_ID_index])
print("game_params ", game_params)

game_param_map = {
	"file_ID": game_params[0],
	"pool": game_params[1], 
	"val_dist": game_params[2],
	"ood1": game_params[3],
	"ood2": game_params[4],
	"o1_pay": game_params[5],
	"o2_pay": game_params[6]
}

te_egta(game_param_map, T, trial_index)



