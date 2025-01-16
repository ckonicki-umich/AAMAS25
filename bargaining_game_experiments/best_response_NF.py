import random
import string
import numpy as np
import math
import itertools as it
import copy
import gc
import shutil
from DQN import *
from bargaining import *

NUM_EVAL_EPISODES = 100

def predict(x, W):
	'''
	'''
	x = x @ W[0] + W[1] # Dense
	x[x<0] = 0 # Relu
	x = x @ W[2] + W[3] # Dense
	x[x<0] = 0 # Relu
	x = x @ W[4] + W[5] # Dense

	return x


def get_state_length(pool):
	'''
	'''
	# represent state features via one-hot encoding
	state_len = 0

	# v
	# 3 item types, anywhere from 0 --> 10 for each type
	state_len += NUM_ITEM_TYPES * math.ceil(math.log(VAL_TOTAL, 2))

	# player o
	# 1 bit, 0 for L and 1 for H
	state_len += 1

	# other player o: 00 for no reveal, 01 for L and 10 for H
	state_len += 2

	# offers over the course of turns, with special notation for
	# "walk" and "deal"
	for i in range(2 * NUM_PLAYER_TURNS):
		# num_items + 1 per item since we're using one-hot encoding, plus 1
		# for the decision to reveal the outside offer or not
		state_len += sum([x + 1 for x in pool]) + 1
		#walk and deal
		state_len += 2

	# final bit for "done"
	state_len += 1

	return state_len

def get_next_state(cur_state, action_history, pool, action, j):
	'''
	'''
	next_state = cur_state[:]
	val_bits = math.ceil(math.log(VAL_TOTAL, 2)) * NUM_ITEM_TYPES

	# 2 indicating player's decision to reveal + signal
	# 1 for other player's outside offer (H/L)
	outside_offer_bits = 3

	pool_size = sum([x + 1 for x in pool]) + 1

	prev_turns_taken = len(action_history) - 4
	num_used_bits = val_bits + outside_offer_bits + prev_turns_taken * (pool_size + 2)
	num_prev_rounds = prev_turns_taken // 2

	# update next_state with player's chosen action via one-hot encoding
	# note: need to account for "walk" and "deal" in addition to the offer space
	one_hot_offer = one_hot_encode_offer(pool, action)
	next_state[num_used_bits:(num_used_bits + pool_size + 2)] = one_hot_offer[:]

	next_history = action_history + (action,)

	if j == 1:
		# we are describing a state for player 1's DQN for BR
		p2_offer_reveal = check_outside_offer_reveal(next_history, 2)
		if p2_offer_reveal:
			o2_bits = None
			o2 = action_history[3]

			if o2 == "H":
				o2_bits = [1, 0]
			else:
				o2_bits = [0, 1]

			next_state[(val_bits + 1):(val_bits + 3)] = o2_bits[:]
	else:
		# we are describing a state for player 2's DQN for BR
		p1_offer_reveal = check_outside_offer_reveal(next_history, 1)
		
		if p1_offer_reveal:
			o1_bits = None
			o1 = action_history[1]

			if o1 == "L":
				o1_bits = [0, 1]
			else:
				o1_bits = [1, 0]

			next_state[:2] = o1_bits[:]

	return next_state

def convert_into_state(action_history, pool):
	'''
	j = {1, 2}
	'''
	v1 = action_history[0]
	o1 = action_history[1]
	v2 = action_history[2]
	o2 = action_history[3]

	j = None
	if len(action_history) % 2 == 0:
		j = 1
	else:
		j = 2

	state_len = get_state_length(pool)
	cur_state = None
	if j == 1:
		oh_v1 = one_hot_encode_valuation(v1)
		oh_o1 = one_hot_encode_outside_offer(o1)
		cur_state = oh_v1 + oh_o1 + [0] * (state_len - len(oh_v1 + oh_o1))

	else:
		oh_v2 = one_hot_encode_valuation(v2)
		oh_o2 = one_hot_encode_outside_offer(o2)
		cur_state = oh_v2 + oh_o2 + [0] * (state_len - len(oh_v2 + oh_o2))

	for action in action_history[4:]:
		next_state = get_next_state(cur_state, action_history, pool, action, j)
		cur_state = next_state[:]

	return cur_state

def get_player_history_given_action_history(action_history):
	'''
	'''
	if len(action_history) % 2 == 0:
		# player 1
		p2_offer_reveal = check_outside_offer_reveal(action_history, 2)
		o2 = tuple()
		if p2_offer_reveal:
			o2 = (action_history[3],)
		# v1, o1, skip v2, maybe include o2
		return action_history[:2] + o2 + action_history[4:]
	else:
		# player 2
		p1_offer_reveal = check_outside_offer_reveal(action_history, 1)
		o1 = tuple()
		if p1_offer_reveal:
			o1 = (action_history[1],)
		# skip v1, maybe include o1, v2, o2
		return o1 + action_history[2:]

def one_hot_encode_valuation(v):
	'''
	@arg (list of int's) v: player valuation for each item in the pool

	Converts a given item's value to an agent into the one-hot format
	'''
	max_bits = math.ceil(math.log(VAL_TOTAL, 2))
	oh_list = []
	for i in range(len(v)):
		str_v = bin(v[i])[2:]
		final_v = "0" * (max_bits - len(str_v)) + str_v
		oh_list += list(final_v)
	
	return [int(x) for x in oh_list]

def one_hot_encode_outside_offer(signal):
	'''
	@arg (str) signal: "H" or "L", representing the player's outside offer being high
		or low

	Converts a given player's outside offer signal into the one-hot format (i.e. a single bit)
	'''
	if signal == "L":
		return [0]
	return [1]

def one_hot_encode_offer(pool, offer):
	'''
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (tuple) offer: partition of the item pool offered by the agent in 
		the format of (player1_share, player2_share) per item

	Converts a partition of the item pool offered by an agent into the one-hot
	format
	'''
	oh_offer = []
	offer_bits = sum([x + 1 for x in pool])
	oo_bit = 0
	if offer[1]:
		oo_bit = 1

	if offer[0] == ("walk",):
		oh_offer = [0] * offer_bits + [oo_bit] + [1] + [0]

	elif offer[0] == ("deal",):
		oh_offer = [0] * offer_bits + [oo_bit] + [0] + [1]

	else:
		
		book_bits = [0] * (pool[0] + 1)
		hat_bits = [0] * (pool[1] + 1)
		ball_bits = [0] * (pool[2] + 1)

		book_i = offer[0][0][0]
		book_bits[book_i] = 1
		oh_offer += book_bits

		hat_i = offer[0][1][0]
		hat_bits[hat_i] = 1
		oh_offer += hat_bits

		ball_i = offer[0][2][0]
		ball_bits[ball_i] = 1
		oh_offer += ball_bits

		oh_offer += [oo_bit]
		oh_offer += [0] * 2

	return oh_offer

def get_offer_given_policy(policy_str, action_space, POLICY_SPACE, true_game_history, pool):
	'''
	@arg (str) policy_str: "pi_" string representing a policy mapping valuations to optimal
		actions/offers in the negotiation in POLICY_SPACE
	@arg (tup of int's) v: player j's valuation for each item in the pool

	Retrieves a corresponding offer in the negotiations given a player's policy string and
	private valuation
	'''
	is_game_start = len(true_game_history) == 4
	policy_weights = POLICY_SPACE.get(policy_str)
	state = convert_into_state(true_game_history, pool)
	best_action = get_best_action(state, policy_weights, action_space, is_game_start)
	offer = best_action[0]

	return offer

def check_outside_offer_reveal(history, player_num):
	'''
	@arg (list) history: history of actions/events that occurred in the game
		leading up to the current node
	@arg (int) player_num: integer indicating which player corresponds to the
		history

	Returns a boolean regarding whether or not the input player chose to reveal his
	outside offer to the other player
	'''
	if len(history) <= 4:
		return False
	
	tuple_actions_only = [x for x in history[4:] if type(x) is tuple]
	for i in range(player_num - 1, len(tuple_actions_only), 2):
		action = tuple_actions_only[i]
		if action == ('deal',) or action == ('walk',):
			return False

		is_bool = type(action[1]) is bool
		if is_bool and action[1]:
			return True

	return False

def get_best_action(state, weights, action_space, is_game_start):
	'''
	'''
	x = np.array(state)
	state_arr = x.reshape(-1, len(state))
	q_output = predict(np.array([state_arr]), weights)
	Qmax_ind = np.argmax(q_output[0])
	best_action = action_space[Qmax_ind]

	if is_game_start and best_action[0] in [("deal",), ("walk",)]:
		action_space_copy = action_space[:-4]
		q_output_copy = q_output[0][0, 0:(q_output[0].size-4)]
		new_Qmax_ind = np.argmax(q_output_copy)
		best_action = action_space_copy[new_Qmax_ind]

	return best_action

def convert_into_best_response_policy(empir_br_infostates, BR_weights, POLICY_SPACE):
	'''
	@arg (map of tup's to maps) BR: best response learned from DQN, mapping each
		player j infoset ID in the empirical game to a corresponding policy given
		explicitly as a map from valuations to actions (pure strat)

	Converts a given set of best response policies for all empirical game information
	sets belonging to a given player, rep'ed as maps, into the same set represented as
	strings in the interest of saving space. Uses global variable POLICY_SPACE that maps
	each ID string to its corresponding policy
	'''
	policy_str = "pi_" + str(len(POLICY_SPACE))
	empir_BR = {}
	for empir_infoset_id in empir_br_infostates:
		history_length = len(empir_infoset_id[1])
		player_id = empir_infoset_id[0]
		signals = [True, False]

		if policy_str not in POLICY_SPACE:
			POLICY_SPACE[policy_str] = BR_weights

		empir_BR[empir_infoset_id] = (policy_str, random.choice(signals))

	return empir_BR

def get_nf_policy_given_action(action, action_space, NF_POLICY_SPACE, true_game_history, pool):
	'''
	'''
	is_game_start = len(true_game_history) == 4
	state = convert_into_state(true_game_history, pool)
	string_prefix = "pi_"

	if is_game_start and action in [("deal",), ("walk",)]:
		raise Exception("Player 1 cannot make a deal or walk on the first turn")

	for policy_str in NF_POLICY_SPACE:
		weights = NF_POLICY_SPACE[policy_str]
		best_action = get_best_action(state, weights, action_space, is_game_start)
		if best_action[0] == action:
			return policy_str

	return None


def get_offer_given_nf_policy(policy_str, action_space, NF_POLICY_SPACE, true_game_history, pool):
	'''
	'''
	is_game_start = len(true_game_history) == 4
	policy_weights = NF_POLICY_SPACE.get(policy_str)
	state = convert_into_state(true_game_history, pool)
	best_action = get_best_action(state, policy_weights, action_space, is_game_start)

	return best_action[0]

def dqn_br_nf_player_1(meta_strategy2, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
	o1_pay_arr, o2_pay_arr, hp_set, NF_POLICY_SPACE1, NF_POLICY_SPACE2, is_regret_eval):
	'''
	@arg (map: tuple --> (map: str --> float)) meta-strategy2: each key
		in the outer map is a player 2 infoset. Each infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: string corresponding to this particular run of TE-EGTA so files can be identified
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (list) o1_pay_arr: 
	@arg (list) o2_pay_arr: 
	@arg (tup) hp_set: list of set hyperparameters in the following order:
		num_training_steps, gamma, epsilon_min, epsilon_annealing, learning_rate, model_width, update_target
	@arg (map) NF_POLICY_SPACE1: 
	@arg (map) NF_POLICY_SPACE2:

	Compute Player 1's best response policy for each infoset/game state in the normal-form representation,
	using a DQN
	'''
	num_training_steps = hp_set["training_steps"]
	trials = int(num_training_steps)
	trial_len = 500
	eval_over_time = []

	# represent state features via one-hot encoding
	state_len = get_state_length(pool)
	BR1 = {}

	# map from player 1 history in true game to the one-hot-encoded state
	# we need this to acquire the player 1 best response
	relevant_p1_states = {}
	offer_space = generate_offer_space(pool)
	action_space = list(it.product(offer_space, [True, False]))
	dqn_agent = DQN((state_len,), action_space, hp_set)
	steps = 0

	for trial in range(trials):
		if trial % 1000 == 0:
			print("BR for player 1, trial # ", trial)

		cur_history = ()
		
		# sample v1 and v2
		v1, v2 = generate_player_valuations(val_dist)
		o1, o2 = generate_player_outside_offers(outside_offer_dist1, outside_offer_dist2)

		# generate payoffs for walking away or failing to reach a consensus
		# with the other agent and instead choosing one's outside offer
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
		cur_history = (v1, o1, v2, o2,)

		cur_state = convert_into_state(cur_history, pool)
		game_start = cur_state[:]

		done = False

		for step in range(trial_len):
			steps += 1
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))

			is_start = False
			if game_start == cur_state:
				is_start = True

			action = dqn_agent.act(cur_state_arr, is_start, steps)

			next_state, reward, done, next_history = player1_step(meta_strategy2, action, cur_history, cur_state, pool, v1, v2, o1, o2, o1_pay, o2_pay, action_space, 
				NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))

			dqn_agent.remember(cur_state_arr, action, reward, next_state_arr, done)

			cur_state = next_state[:]
			cur_history = next_history

			avg_q_output = dqn_agent.replay(steps)
			if done:
				break
			
			if steps >= num_training_steps:
				break

		# same for outer loop
		if steps >= num_training_steps:
			break

		# if trial % 200 == 0:
		# 	br1_pay = evaluate_nf_player1(dqn_agent, meta_strategy2, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
		# 		NF_POLICY_SPACE1, NF_POLICY_SPACE2)
		# 	eval_over_time.append(br1_pay)
		# 	#print("BR1_pay ", br1_pay, " for trial # ", trial)


	final_model_name = "success_1_NF_" + file_ID + ".model"
	dqn_agent.save_model(final_model_name)
	reconstructed_model = tf.keras.models.load_model(final_model_name)
	BR1_weights = reconstructed_model.get_weights()

	del dqn_agent
	shutil.rmtree(final_model_name)
	del reconstructed_model
	gc.collect()

	if is_regret_eval:
		return BR1_weights, NF_POLICY_SPACE1

	else:
		policy_str = "pi_" + str(len(NF_POLICY_SPACE1))
		if policy_str not in NF_POLICY_SPACE1:
			NF_POLICY_SPACE1[policy_str] = BR1_weights

		return (policy_str, random.choice([True, False])), NF_POLICY_SPACE1

def player1_step(meta_strategy2, p1_action, cur_history, cur_state, pool, v1, v2, o1, o2, o1_pay, o2_pay, p2_actions, NF_POLICY_SPACE1, 
	NF_POLICY_SPACE2):
	'''
	@arg (map: tuple --> (map: str --> float)) meta_strategy: Current metastrategy. Each key in the 
		outer map is a player infoset. Each player infoset's strategy is represented as a second map 
		giving a distribution over that infoset's available policies
	@arg (str) p1_action: Chosen action of player 1 to be played out
	@arg (tuple) cur_history: History corresponding to the current player 1 node in the game
	@arg (list of ints (0/1)) cur_state: One-hot encoding of cur_history
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (tuple of int's) v1: player 1's valuation for each item in the pool
	@arg (tuple of int's) v2: player 2's valuation for each item in the pool
	@arg (str) o1: player 1's outside offer signal
	@arg (str) o2: player 2's outside offer signal
	@arg (int) o1_pay: payoff to player 1 for accepting its private outside offer
	@arg (int) o2_pay: payoff to player 2 for accepting its private outside offer
	@arg (list of tup's) p2_actions: action space for player 2

	Steps through true game environment given the current state and player 1's chosen action; returns
	the next state, reward (if any), and updated history. Corresponds to env.step() function one might
	find when applying DQNs to a gym env
	'''
	next_history = cur_history + (p1_action,)

	next_state = get_next_state(cur_state, cur_history, pool, p1_action, 1)
	prev_turns_taken = len(cur_history) - 4
	num_prev_rounds = prev_turns_taken // 2

	offer_space = generate_offer_space(pool)
	action_space = list(it.product(offer_space, [True, False]))
	
	reward = 0
	w = []
	a_space = []
	done = False

	# check if we're at the end of the game, meaning player 1 chose deal or walk
	if p1_action[0] == ("walk",) or p1_action[0] == ("deal",):
		reward = compute_utility(p1_action[0], pool, v1, v2, cur_history[-1][0], o1_pay, o2_pay, num_prev_rounds)[0]
		done = True
		next_state[-1] = 1

	else:
		for empir_action2 in meta_strategy2.keys():
			a_space.append(empir_action2)
			w.append(meta_strategy2.get(empir_action2))

		p2_empir_action = random.choices(a_space, weights=w)[0]
		policy_str = p2_empir_action[0]
		p2_offer = get_offer_given_nf_policy(p2_empir_action[0], action_space, NF_POLICY_SPACE2, next_history, pool)
		cur_history = next_history

		p2_action = (p2_offer, bool(p2_empir_action[1]))
		next_history = cur_history + (p2_action,)
		next_state = get_next_state(next_state[:], cur_history, pool, p2_action, 1)

		# check if we're at the end of the game, meaning the number of turns is up, or player 2
		# chose deal or walk
		if p2_action[0] == ("walk",) or p2_action[0] == ("deal",):
			reward = compute_utility(p2_action[0], pool, v1, v2, cur_history[-1][0], o1_pay, o2_pay, num_prev_rounds)[0]
			done = True
			next_state[-1] = 1
		elif num_prev_rounds + 1 == NUM_PLAYER_TURNS:
			reward = compute_utility(("walk",), pool, v1, v2, cur_history[-1][0], o1_pay, o2_pay, NUM_PLAYER_TURNS)[0]
			done = True
			next_state[-1] = 1

	return next_state, reward, done, next_history

def evaluate_nf_player1(dqn_agent, meta_strategy2, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, NF_POLICY_SPACE1, NF_POLICY_SPACE2):
	'''
	@arg (DQN) dqn_agent: agent for DQN representing player 1
	@arg (map: tuple --> (map: str --> float)) meta-strategy: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: string corresponding to this particular run of TE-EGTA so files can be identified
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")

	Evaluates the average reward over a series of episodes (simulated gameplay) to player 1 given the 
	now-trained DQN
	'''
	total_reward_over_time = 0
	trial_len = 500

	offer_space = generate_offer_space(pool)
	action_space = list(it.product(offer_space, [True, False]))

	for ep in range(NUM_EVAL_EPISODES):
		if ep % 100 == 0:
			print("Evaluation for player 1, episode # ", ep)
		
		cur_history = ()
		# sample v1 and v2
		v1, v2 = generate_player_valuations(val_dist)
		o1, o2 = generate_player_outside_offers(outside_offer_dist1, outside_offer_dist2)

		# generate payoffs for walking away or failing to reach a consensus
		# with the other agent and instead choosing one's outside offer
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

		cur_history = (v1, o1, v2, o2,)
		cur_state = convert_into_state(cur_history, pool)
		game_start = cur_state[:]
		done = False

		for step in range(trial_len):
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))

			is_start = False
			if game_start == cur_state:
				is_start = True

			action = dqn_agent.act_in_eval(cur_state_arr, is_start)
			next_state, reward, done, next_history = player1_step(meta_strategy2, action, cur_history, cur_state, pool, 
				v1, v2, o1, o2, o1_pay, o2_pay, action_space, NF_POLICY_SPACE1, NF_POLICY_SPACE2)
			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))

			cur_state = next_state[:]
			cur_history = next_history
			
			if done:
				total_reward_over_time += reward
				break

	return float(total_reward_over_time) / NUM_EVAL_EPISODES

def dqn_br_nf_player_2(meta_strategy1, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, hp_set, 
	NF_POLICY_SPACE1, NF_POLICY_SPACE2, is_regret_eval):
	'''
	@arg (map: tuple --> (map: str --> float)) meta-strategy1: each key
		in the outer map is a player 1 infoset. Each infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: string corresponding to this particular run of TE-EGTA so files can be identified
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (list) o1_pay_arr: 
	@arg (list) o2_pay_arr: 
	@arg (tup) hp_set: list of set hyperparameters in the following order:
		num_training_steps, gamma, epsilon_min, epsilon_annealing, learning_rate, model_width, update_target
	@arg (map) NF_POLICY_SPACE1: 
	@arg (map) NF_POLICY_SPACE2:

	Compute Player 2's best response policy for each infoset/game state in the normal-form representation,
	using a DQN
	'''
	num_training_steps = hp_set["training_steps"]
	trial_len = 500
	trials = int(num_training_steps)

	# represent state features via one-hot encoding
	state_len = get_state_length(pool)
	eval_over_time = []
	BR2 = {}

	# map from player 2 history in true game to the one-hot-encoded state
	# we need this to acquire the player 2 best response
	relevant_p2_states = {}
	offer_space = generate_offer_space(pool)
	action_space = list(it.product(offer_space, [True, False]))
	dqn_agent = DQN((state_len,), action_space, hp_set)
	steps = 0

	for trial in range(trials):
		if trial % 1000 == 0:
			print("BR for player 2, trial # ", trial)
		
		cur_history = ()
		# sample v1 and v2
		v1, v2 = generate_player_valuations(val_dist)
		o1, o2 = generate_player_outside_offers(outside_offer_dist1, outside_offer_dist2)

		# generate payoffs for walking away or failing to reach a consensus
		# with the other agent and instead choosing one's outside offer
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)

		cur_history = cur_history + (v1, o1, v2, o2,)
		cur_state = convert_into_state(cur_history, pool)
		
		a_space = []
		w = []
		for empir_action1 in meta_strategy1.keys():
			a_space.append(empir_action1)
			w.append(meta_strategy1.get(empir_action1))

		p1_empir_action = random.choices(a_space, weights=w)[0]
		p1_offer = get_offer_given_nf_policy(p1_empir_action[0], action_space, NF_POLICY_SPACE1, cur_history, pool)
		p1_action = (p1_offer, bool(p1_empir_action[1]))
		if p1_offer in [("deal",), ("walk",)]:
			break

		prev_turns_taken = len(cur_history) - 4
		next_history = cur_history + (p1_action,)
		next_state = get_next_state(cur_state, cur_history, pool, p1_action, 2)
		assert len(next_state) == state_len

		cur_state = next_state[:]
		game_start = cur_state[:]
		cur_history = next_history
		done = False

		for step in range(trial_len):
			steps += 1
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))
			is_start = False

			if game_start == cur_state:
				is_start = True

			action = dqn_agent.act(cur_state_arr, is_start, step)
			assert len(cur_state) == state_len
			next_state, reward, done, next_history = player2_step(meta_strategy1, action, cur_history, cur_state, pool, v1, v2, o1, o2, o1_pay, o2_pay, action_space, NF_POLICY_SPACE1,
				NF_POLICY_SPACE2)

			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))
			assert len(next_state) == state_len
			dqn_agent.remember(cur_state_arr, action, reward, next_state_arr, done)
			cur_state = next_state[:]
			cur_history = next_history
			avg_q_output = dqn_agent.replay(steps)

			if done:
				break

			if steps >= num_training_steps:
				break

		if steps >= num_training_steps:
			break

		# if trial % 200 == 0:
		# 	br2_pay = evaluate_nf_player2(dqn_agent, meta_strategy1, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
		# 		NF_POLICY_SPACE1, NF_POLICY_SPACE2)
		# 	eval_over_time.append(br2_pay)
		# 	print("BR2_pay ", br2_pay, " for trial # ", trial)

	final_model_name = "success_2_NF_" + file_ID + ".model"
	dqn_agent.save_model(final_model_name)
	reconstructed_model = tf.keras.models.load_model(final_model_name)
	BR2_weights = reconstructed_model.get_weights()

	del dqn_agent
	del reconstructed_model
	gc.collect()
	shutil.rmtree(final_model_name)

	if is_regret_eval:
		return BR2_weights, NF_POLICY_SPACE2
	else:
		policy_str = "pi_" + str(len(NF_POLICY_SPACE2))
		if policy_str not in NF_POLICY_SPACE2:
			NF_POLICY_SPACE2[policy_str] = BR2_weights

		return (policy_str, random.choice([True, False])), NF_POLICY_SPACE2

def player2_step(meta_strategy1, p2_action, cur_history, cur_state, pool, v1, v2, o1, o2, o1_pay, o2_pay, p1_actions, NF_POLICY_SPACE1, NF_POLICY_SPACE2):
	'''
	@arg (map: tuple --> (map: str --> float)) meta_strategy: Current metastrategy. Each key in the 
		outer map is a player infoset. Each player infoset's strategy is represented as a second map 
		giving a distribution over that infoset's action space
	@arg (str) p2_action: Chosen action of player 2 to be played out
	@arg (tuple) cur_history: History corresponding to the current player 2 node in the game
	@arg (list of ints (0/1)) cur_state: One-hot encoding of cur_history
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (tuple of int's) v1: player 1's valuation for each item in the pool
	@arg (tuple of int's) v2: player 2's valuation for each item in the pool
	@arg (str) o1: player 1's outside offer signal
	@arg (str) o2: player 2's outside offer signal
	@arg (int) o1_pay: payoff to player 1 for accepting its private outside offer
	@arg (int) o2_pay: payoff to player 2 for accepting its private outside offer
	@arg (list of tup's) p1_actions: action space for player 1

	Steps through true game environment given the current state and player 2's chosen action; returns
	the next state, reward (if any), and updated history. Corresponds to env.step() function one might
	find when applying DQNs to a gym env
	'''
	next_history = cur_history + (p2_action,)
	next_state = get_next_state(cur_state, cur_history, pool, p2_action, 2)
	prev_turns_taken = len(cur_history) - 4
	num_prev_rounds = prev_turns_taken // 2
	reward = 0
	w = []
	a_space = []
	done = False
	offer_space = generate_offer_space(pool)
	action_space = list(it.product(offer_space, [True, False]))

	if p2_action[0] == ("walk",) or p2_action[0] == ("deal",):
		reward = compute_utility(p2_action[0], pool, v1, v2, cur_history[-1][0], o1_pay, o2_pay, num_prev_rounds)[1]
		done = True
		next_state[-1] = 1

	# check if the number of turns is up
	elif num_prev_rounds + 1 == NUM_PLAYER_TURNS:
		reward = compute_utility(("walk",), pool, v1, v2, cur_history[-1][0], o1_pay, o2_pay, NUM_PLAYER_TURNS)[1]
		done = True
		next_state[-1] = 1
	else:

		for empir_action1 in meta_strategy1.keys():
			a_space.append(empir_action1)
			w.append(meta_strategy1.get(empir_action1))

		p1_empir_action = random.choices(a_space, weights=w)[0]
		p1_offer = get_offer_given_nf_policy(p1_empir_action[0], action_space, NF_POLICY_SPACE1, next_history, pool)
		p1_action = (p1_offer, p1_empir_action[1])
		cur_history = next_history
		next_history = cur_history + (p1_action,)
		cur_state = next_state[:]
		next_state = get_next_state(cur_state, cur_history, pool, p1_action, 2)

		# check if we're at the end of the game, meaning player 1 chose deal or walk
		if p1_action[0] == ("walk",) or p1_action[0] == ("deal",):
			reward = compute_utility(p1_action[0], pool, v1, v2, cur_history[-1][0], o1_pay, o2_pay, num_prev_rounds)[1]
			done = True
			next_state[-1] = 1

	return next_state, reward, done, next_history

def evaluate_nf_player2(dqn_agent, meta_strategy1, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, 
	NF_POLICY_SPACE1, NF_POLICY_SPACE2, default_policy1, default_policy2):
	'''
	@arg (DQN) dqn_agent: agent for DQN representing player 1
	@arg (map: tuple --> (map: str --> float)) meta-strategy: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: string corresponding to this particular run of TE-EGTA so files can be identified
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (list) o1_pay_arr: 
	@arg (list) o2_pay_arr: 
	@arg (map) NF_POLICY_SPACE1: 
	@arg (map) NF_POLICY_SPACE2:

	Evaluates the average reward over a series of episodes (simulated gameplay) to player 2 given the 
	now-trained DQN
	'''
	trial_len = 500
	total_reward_over_time = 0

	# represent state features via one-hot encoding
	state_len = get_state_length(pool)
	offer_space = generate_offer_space(pool)
	action_space = list(it.product(offer_space, [True, False]))
	
	for ep in range(NUM_EVAL_EPISODES):
		if ep % 100 == 0:
			print("Evaluation for player 2, episode # ", ep)
		
		cur_history = ()
		# sample v1 and v2
		v1, v2 = generate_player_valuations(val_dist)
		o1, o2 = generate_player_outside_offers(outside_offer_dist1, outside_offer_dist2)
		
		# generate payoffs for walking away or failing to reach a consensus
		# with the other agent and instead choosing one's outside offer
		o1_pay = get_pay_given_outside_offer(o1_pay_arr, o1)
		o2_pay = get_pay_given_outside_offer(o2_pay_arr, o2)
		cur_history = (v1, o1, v2, o2,)
		cur_state = convert_into_state(cur_history, pool)
		a_space = []
		w = []

		for empir_action1 in meta_strategy1.keys():
			a_space.append(empir_action1)
			w.append(meta_strategy1.get(empir_action1))

		p1_empir_action = random.choices(a_space, weights=w)[0]
		p1_offer = get_offer_given_nf_policy(p1_empir_action[0], action_space, NF_POLICY_SPACE1, cur_history, pool)
		p1_action = (p1_offer, bool(p1_empir_action[1]))

		if p1_offer in [("deal",), ("walk",)]:
			break

		prev_turns_taken = len(cur_history) - 4
		next_history = cur_history + (p1_action,)
		next_state = get_next_state(cur_state, cur_history, pool, p1_action, 2)
		assert len(next_state) == state_len

		cur_state = next_state[:]
		game_start = cur_state[:]
		cur_history = next_history
		done = False

		for step in range(trial_len):
			steps += 1
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))
			is_start = False

			if game_start == cur_state:
				is_start = True

			action = dqn_agent.act(cur_state_arr, is_start, step)
			assert len(cur_state) == state_len
			next_state, reward, done, next_history = player2_step(meta_strategy1, action, policy_str, cur_history, cur_state, pool, v1, v2, o1, o2, o1_pay, o2_pay, action_space, NF_POLICY_SPACE1)
			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))
			assert len(next_state) == state_len

			dqn_agent.remember(cur_state_arr, action, reward, next_state_arr, done)
			cur_state = next_state[:]
			cur_history = next_history
			avg_q_output = dqn_agent.replay(steps)

			if done:
				total_reward_over_time += reward
				break

	return float(total_reward_over_time) / NUM_EVAL_EPISODES


def compute_best_response_NF(meta_strategy1, meta_strategy2, file_ID, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
	o1_pay_arr, o2_pay_arr, hp_set1, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2, is_regret_eval):
	'''
	@arg (map: tuple --> (map: str --> float)) meta-strategy1: each key
		in the outer map is a player 1 infoset/state in the true game. Each infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (map: tuple --> (map: str --> float)) meta-strategy2: each key
		in the outer map is a player 2 infoset/state in the true game. Each infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: identification string for file containing outputs (error, regret, etc.)
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool
	@arg (dict) outside_offer_dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) outside_offer_dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (list) o1_pay_arr: 
	@arg (list) o2_pay_arr: 
	@arg (tup) hp_set1: list of set hyperparameters for player 1's DQN in the following order:
		num_training_steps, gamma, epsilon_min, epsilon_annealing, learning_rate, model_width, update_target
	@arg (tup) hp_set2: list of set hyperparameters for player 2's DQN in the following order:
		num_training_steps, gamma, epsilon_min, epsilon_annealing, learning_rate, model_width, update_target
	@arg (map) NF_POLICY_SPACE1: 
	@arg (map) NF_POLICY_SPACE2:

	Compute each player's best response for each infoset without the true game ExtensiveForm object,
	using tabular Q-learning
	'''
	BR1 = {}
	BR2 = {}

	# BR1, br1_payoffs_over_time, NF_POLICY_SPACE1 = dqn_br_nf_player_1(meta_strategy2, file_ID, pool, val_dist, outside_offer_dist1, 
	# 	outside_offer_dist2, o1_pay_arr, o2_pay_arr, hp_set1, NF_POLICY_SPACE1, NF_POLICY_SPACE2, is_regret_eval)
	BR1, NF_POLICY_SPACE1 = dqn_br_nf_player_1(meta_strategy2, file_ID, pool, val_dist, outside_offer_dist1, 
		outside_offer_dist2, o1_pay_arr, o2_pay_arr, hp_set1, NF_POLICY_SPACE1, NF_POLICY_SPACE2, is_regret_eval)

	# BR2, br2_payoffs_over_time, NF_POLICY_SPACE2 = dqn_br_nf_player_2(meta_strategy1, file_ID, pool, val_dist, outside_offer_dist1, 
	# 	outside_offer_dist2, o1_pay_arr, o2_pay_arr, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2, is_regret_eval)
	BR2, NF_POLICY_SPACE2 = dqn_br_nf_player_2(meta_strategy1, file_ID, pool, val_dist, outside_offer_dist1, 
		outside_offer_dist2, o1_pay_arr, o2_pay_arr, hp_set2, NF_POLICY_SPACE1, NF_POLICY_SPACE2, is_regret_eval)
	
	# return BR1, BR2, br1_payoffs_over_time, br2_payoffs_over_time, NF_POLICY_SPACE1, NF_POLICY_SPACE2
	return BR1, BR2, NF_POLICY_SPACE1, NF_POLICY_SPACE2


