import random
import string
import numpy as np
import math
import itertools as it
import copy
import gc
import shutil
import sys
from DQN import *
from bargaining import *
from best_response import *

# I = 0
random.seed(0)

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
'BIG_DoND_TY43L', 
'BIG_DoND_9H7NI', 
'BIG_DoND_RBINZ', 
'BIG_DoND_JW5X5'
]

# file_ID = file_ID_list[I]
# print("file_ID ", file_ID)


'''
3-Phase Hyperparameter Tuning of DQNs for Best Response
'''
grid_space = {
	"epsilon_min": [0.01, 0.02, 0.05],
	"epsilon_annealing": ["linear", "exp"],
	"model_width": [50, 100, 200],
	"gamma": [0.99],
	"update_target": [1, 2, 5],
	"learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
	"training_steps": [5e5, 1e6, 1.5e6, 2e6]
	#"training_steps": [5e3]
}

grid_space_phase1 = {
	# in grid
	"epsilon_annealing": ["linear", "exp"],
	"training_steps": [5e5, 1e6, 1.5e6, 2e6],
	#"training_steps": [5e3],
	"learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
	"update_target": [1, 2, 5],
	# single
	"gamma": [0.99],
	"model_width": [100],
	"epsilon_min": [0.02], 
}

'''
Phase 1:
	- training_steps
	- epsilon_annealing
	- learning_rate
	- update_target


Phase 2:
	- gamma
	- epsilon_min
	- model_width
'''

def dict_product(d):
	'''
	@arg (dict) d: dictionary representing the grid space of hyperparameters and
		their correpsonding value ranges

	Helper method to generate each combination of hyperparameter values
	'''
	keys = d.keys()
	hp_set_list = []
	count = 0
	for elt in it.product(*d.values()):
		#yield dict(zip(keys, elt))
		print(count)
		count += 1
		hp_set_list.append(dict(zip(keys, elt)))
		
	np.savez_compressed("all_hp_sets.npz", np.array(hp_set_list))

	return None

def retrieve_game(file_ID_index):
	'''
	@arg (int) file_ID_index: index corresponding to the game under consideration

	Helper method allowing us to iterate over each sequential bargaining game
	for hyperparameter tuning
	'''
	a_f = np.load("game_parameters.npz", allow_pickle=True)
	lst = a_f.files
	for params in a_f['arr_0']:
		print("params ", params)
		print("\n")
		if file_ID_index == params[0]:
			return params
		

def tuning_br(grid_space, game_param_map, hp_set_index):
	'''
	@arg (dict) grid_space: Grid space of hyperparameters, mapped to singleton lists (if not in current 
		tuning phase) or non-singleton lists (if in current tuning phase)
	@arg (dict) game_param_map: Map of each input parameter type to the corresponding values
		for the given game
	@arg (int) hp_set_index: index corresponding to the set of values (out of all possible
		combinations) assigned to each hyperparameter

	Tune hyperparameters within phase 2 (gamma, epsilon_min, model_width) for a given game 
	for the DQNs learning the best response
	'''

	num_trials = 5
	file_ID = game_param_map["file_ID"]
	pool = game_param_map["pool"]
	val_dist = game_param_map["val_dist"]
	outside_offer_dist1 = game_param_map["ood1"]
	outside_offer_dist2 = game_param_map["ood2"]
	o1_pay_arr = game_param_map["o1_pay"]
	o2_pay_arr = game_param_map["o2_pay"]

	# metric for performance of player 1
	performance_player1_hp = {}

	# metric for performance of player 2
	performance_player2_hp = {}

	#for hp_set in dict_product(grid_space):
	hp_f = np.load("all_hp_sets.npz", allow_pickle=True)

	hp_set = hp_f['arr_0'][hp_set_index]
	print(hp_set)
	#print(assdsd)
	print("hp set ", hp_set)
	performance1 = []
	performance2 = []

	# set random meta_strategy for player 1
	meta_strategy1 = {}

	# set random meta_strategy for player 2
	meta_strategy2 = {}

	network_ID = file_ID + "_" + str(hp_set_index)

	POLICY_SPACE1 = {}
	POLICY_SPACE2 = {}

	#np.save(network_ID + "_policymap1.npy", POLICY_SPACE1)
	#np.save(network_ID + "_policymap2.npy", POLICY_SPACE2)

	for i in range(num_trials):
		print("trial num ", i)

		if i > 0:
			print("i ", i)
			print("can load")
			POLICY_SPACE1 = np.load(network_ID + "_policymap1.npy", allow_pickle=True).item()
			POLICY_SPACE2 = np.load(network_ID + "_policymap2.npy", allow_pickle=True).item()
		
		br1, payoff1_over_time, avg1, POLICY_SPACE1 = dqn_br_player_1(meta_strategy2, network_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, hp_set, POLICY_SPACE1, POLICY_SPACE2)

		# update performance1 with player 1 payoff from playing BR1
		performance1.append(avg1)
		np.save(network_ID + "_policymap1.npy", POLICY_SPACE1)



		br2, payoff2_over_time, avg2, POLICY_SPACE2 = dqn_br_player_2(meta_strategy1, network_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, hp_set, POLICY_SPACE1, POLICY_SPACE2)

		# update performance2 with average player 2 payoff from playing BR2
		performance2.append(avg2)
		print("avg2 ", avg2)

		np.save(network_ID + "_policymap2.npy", POLICY_SPACE2)

		for x in br1.keys():
			meta_strategy1[x] = {br1[x]: 1.0}
		#meta_strategy1 = br1.copy()
		for x in br2.keys():
			meta_strategy2[x] = {br2[x]: 1.0}
		#meta_strategy2 = br2.copy()

	print("perform1 ", performance1)
	print("perform2 ", performance2)

	performance_player1_hp[str(hp_set)] = np.mean(performance1)

	performance_player2_hp[str(hp_set)] = np.mean(performance2)
	#print(stahp)


	np.savez_compressed("hp_tuning_" + file_ID + "_" + str(hp_set_index), performance_player1_hp, performance_player2_hp)


file_ID_index = int(sys.argv[1])
random_index = int(sys.argv[2])

print(file_ID_index)
print(random_index)

file_ID = file_ID_list[file_ID_index]
print("file_ID ", file_ID)
# print(aaaa)

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

# randomly generated once
#random_indices = np.random.randint(119, size=(20))
random_indices = np.array([ 16,  70,  34,  86,  71,  26,  20, 112, 118,   7,  74,  39,  89,
        17,  95,  21, 106,  19,  58,  95])
hp_set_index = random_indices[random_index]

tuning_br(grid_space_phase1, game_param_map, hp_set_index)
