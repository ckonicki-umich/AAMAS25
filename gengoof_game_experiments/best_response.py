import random
import string
import numpy as np
import math
import itertools as it
import copy
import gc
import shutil
from DQN import *
from abstract_games import *

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

def softmax(x):
	'''
	'''
	tau = 1.0
	exp_values = [np.exp(i / tau) for i in x]
	exp_values_sum = np.sum(exp_values)
	if exp_values_sum == 0.0:
		return np.array([1.0 / len(x) for i in x])

	return exp_values / exp_values_sum

def get_state_length(num_rounds):
	'''
	'''
	# represent state features via one-hot encoding
	state_len = 0

	# redid with a larger state shape to encompass more features via one-hot encoding
	p1_rounds = num_rounds * (num_rounds - 1)
	p2_rounds = num_rounds * (num_rounds - 1)

	# sum_{i = 1}^num_rounds i ... minus 1
	chance_rounds = num_rounds * (num_rounds - 1)

	# final bit for "done"
	state_len = p1_rounds + p2_rounds + chance_rounds + 1

	return state_len

def convert_into_state(action_history, num_rounds, p1_actions, p2_actions, chance_events):
	'''
	'''
	state_len = get_state_length(num_rounds)
	cur_state = []
	j = None
	if len(action_history) % 3 == 1:
		j = 1
	elif len(action_history) % 3 == 2:
		j = 2
	else:
		j = 0

	for i in range(0, len(action_history), 3):
		chance_outcome = action_history[i]
		oh_chance_outcome = one_hot_encode_chance_outcome(chance_outcome, chance_events)
		cur_state += oh_chance_outcome

		if i + 1 < len(action_history):
			oh_action = one_hot_encode_player_action(action_history[i+1], p1_actions)
			
			# Player 2 isn't supposed to know what player 1 did in the current round
			if j == 2 and i + 1 >= (len(action_history) - 1):
				continue
			else:
				cur_state += oh_action

		if i + 2 < len(action_history):
			oh_action = one_hot_encode_player_action(action_history[i+2], p2_actions)
			cur_state += oh_action
	
	cur_state += [0] * (state_len - len(cur_state))
	
	return cur_state

def one_hot_encode_chance_outcome(outcome, chance_events):
	'''
	'''
	oh_outcome = [0] * len(chance_events)
	event_index = chance_events.index(outcome)
	oh_outcome[event_index] = 1
	return oh_outcome

def one_hot_encode_player_action(action, player_actions):
	'''
	'''
	oh_action = [0] * len(player_actions)
	action_index = player_actions.index(action)
	oh_action[action_index] = 1
	return oh_action

def get_action_given_policy(policy_str, action_space, POLICY_SPACE, true_game_history, game_params):
	'''
	'''
	num_rounds, included_rounds, p1_actions, p2_actions, chance_events = game_params
	policy_weights = POLICY_SPACE.get(policy_str)
	state = convert_into_state(true_game_history, num_rounds, p1_actions, p2_actions, chance_events)
	best_action = get_best_action(state, policy_weights, action_space)

	return best_action

def get_policy_given_action(action, action_space, POLICY_SPACE, true_game_history, game_params):
	'''
	'''
	num_rounds, included_rounds, p1_actions, p2_actions, chance_events = game_params
	state = convert_into_state(true_game_history, num_rounds, p1_actions, p2_actions, chance_events)
	string_prefix = "pi_"
	for policy_str in POLICY_SPACE:
		policy_weights = POLICY_SPACE[policy_str]
		best_action = get_best_action(state, policy_weights, action_space)

		if best_action == action:
			return policy_str

	return None

def get_best_action(state, weights, action_space):
	'''
	'''
	x = np.array(state)
	state_arr = x.reshape(-1, len(state))
	q_output = predict(np.array([state_arr]), weights)
	Qmax_ind = np.argmax(q_output[0])
	best_action = action_space[Qmax_ind]

	return best_action

def get_empirical_infoset_id_given_history(action_history, game_params, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	num_rounds, included_rounds, p1_actions, p2_actions, chance_events = game_params
	empir_history = []
	player_id = None

	for i in range(len(action_history)):
		round_num = i // 3
		a = action_history[i]

		if i % 3 == 0:
			# chance event
			if round_num in included_rounds:
				empir_history.append(a)

		elif i % 3 == 1:
			# player 1 action
			policy_str = get_policy_given_action(a, p1_actions, POLICY_SPACE1, action_history[:i], game_params)
			
			if round_num in included_rounds:
				empir_history.append(a)
			else:
				policy_str = get_policy_given_action(a, p1_actions, POLICY_SPACE1, action_history[:i], game_params)
				empir_history.append(policy_str)
			

		else:
			policy_str = get_policy_given_action(a, p2_actions, POLICY_SPACE2, action_history[:i], game_params)
			
			if round_num in included_rounds:
				empir_history.append(a)
			else:
				policy_str = get_policy_given_action(a, p2_actions, POLICY_SPACE2, action_history[:i], game_params)
				empir_history.append(policy_str)
			
	player_num = len(action_history) % 3
	
	return get_empirical_infoset_id_given_empir_history(empir_history, player_num)

def get_empirical_infoset_id_given_empir_history(empir_history, player_num):
	'''
	'''
	if player_num == 1:
		return (1, tuple(empir_history))
	else:
		return (2, tuple(empir_history[:-1]))

def get_round_number(empir_infoset_id, chance_events, included_rounds, num_rounds):
	'''
	'''
	empir_history_len = len(empir_infoset_id[1])
	num_chance_events = len([e for e in empir_infoset_id[1] if e in chance_events])
	x = 0
	
	for r in range(num_rounds - 1):
		if r in included_rounds:
			x += 3
		else:
			x += 2

		if x > empir_history_len:
			return r

def get_last_policy_str(last_action, true_history, action_space, PS, game_params):
	'''
	'''
	input_history = true_history[:-3]
	last_policy_str = get_policy_given_action(last_action, action_space, PS, input_history, game_params)
	index = -3

	while last_policy_str is None and -1 * (index - 3) < len(true_history):
		last_action = true_history[index - 3]
		input_history = true_history[:(index - 3)]
		last_policy_str = get_policy_given_action(last_action, action_space, PS, input_history, game_params)
		index -= 3

	return last_policy_str


def convert_into_best_response_policy(empir_br_infostates, policy_str, BR_weights, game_param_map):
	'''
	'''
	included_rounds = game_param_map["included_rounds"]
	chance_events = game_param_map["chance_events"]
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]

	empir_BR = {}
	for empir_infoset_id in empir_br_infostates:
		action_space = None
		
		if empir_infoset_id[0] == 1:
			action_space = p1_actions[:]
		else:
			assert empir_infoset_id[0] == 2
			action_space = p2_actions[:]

		round_num = get_round_number(empir_infoset_id, chance_events, included_rounds, num_rounds)
		
		if round_num in included_rounds:
			state = convert_into_state(empir_infoset_id[1], num_rounds, p1_actions, p2_actions, chance_events)
			best_action = get_best_action(state, BR_weights, action_space)
			empir_BR[empir_infoset_id] = best_action
		else:
			empir_BR[empir_infoset_id] = policy_str

	return empir_BR


def dqn_br_player_1(meta_strategy, mss_type, file_ID, game_param_map, hp_set, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, is_regret_eval):
	'''
	@arg (map: Infoset --> (map: str --> float)) meta-strategy: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) network_ID: string corresponding to this particular network for training
	@arg (int) num_rounds: number of rounds R
	@arg (list of ints) included_rounds: list of rounds between 0 and R - 1 whose stochastic events we wish to 
		include in the empirical game model
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game
	@arg (map) coarse_to_true_histories_map: Map of each coarsened infoset in empirical game to corr. infoset in true game
	@arg (dict) hp_set: map of set hyperparameters with the following keys:
		num_training_steps, gamma, epsilon_min, epsilon_annealing, learning_rate, model_width, update_target

	Compute Player 1's best response for each infoset without the true game ExtensiveForm object,
	using a DQN
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]
	
	num_training_steps = hp_set["training_steps"]
	trials = int(num_training_steps)
	trial_len = 500
	eval_over_time = []
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	# final bit for "done"
	state_len = get_state_length(num_rounds)
	action_space = p1_actions[:]
	dqn_agent = DQN((state_len,), action_space, hp_set)
	steps = 0
	
	for trial in range(trials):
		if trial % 1000 == 0:
			print("BR for player 1, trial # ", trial)

		cur_history = ()
		empir_history = ()
		outcome, event_index = sample_stochastic_event_given_history(cur_history, chance_events, card_weights)
		cur_history = cur_history + (outcome,)
		empir_history = empir_history + (outcome,)
		
		done = False
		cur_state = convert_into_state(cur_history, num_rounds, p1_actions, p2_actions, chance_events)

		for step in range(trial_len):
			steps += 1
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))
			action = dqn_agent.act(cur_state_arr, steps)
			empir_policy = None
			round_num = math.floor(len(cur_history) / 3)
			
			if round_num in included_rounds:
				empir_policy = action
			else:
				empir_policy = get_policy_given_action(action, p1_actions, POLICY_SPACE1, cur_history, game_params)
			
			next_state, reward, done, next_history, next_empir_history = player1_step(meta_strategy, action, empir_policy, cur_history, empir_history, cur_state, game_param_map, POLICY_SPACE1, POLICY_SPACE2, 
				default_policy1, default_policy2)

			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))
			assert len(next_state) == state_len
			dqn_agent.remember(cur_state_arr, action, reward, next_state_arr, done)

			cur_state = next_state[:]
			cur_history = next_history
			empir_history = next_empir_history
			avg_q_output = dqn_agent.replay(steps)
			
			if done:
				break

			if steps >= num_training_steps:
				break


		if steps >= num_training_steps:
			break

		'''
		if trial % 200 == 0:
			br1_pay = evaluate_player1(dqn_agent, meta_strategy, file_ID, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2)
			eval_over_time.append(br1_pay)
			# print("BR1_pay ", br1_pay, " for trial # ", trial)
		'''
	
	final_model_name = "success_1_" + file_ID + "_" + str(included_rounds[-1]) + "_" + mss_type + ".model"
	dqn_agent.save_model(final_model_name)
	reconstructed_model = tf.keras.models.load_model(final_model_name)
	BR1_weights = reconstructed_model.get_weights()
		
	del dqn_agent
	shutil.rmtree(final_model_name)
	del reconstructed_model
	gc.collect()

	if is_regret_eval:
		return BR1_weights, POLICY_SPACE1

	#return BR1_weights, eval_over_time, POLICY_SPACE1
	return BR1_weights, POLICY_SPACE1


def dqn_br_player_2(meta_strategy, mss_type, file_ID, game_param_map, hp_set, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, is_regret_eval):
	'''
	@arg (map: Infoset --> (map: str --> float)) meta-strategy: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) network_ID: string corresponding to this particular network for training
	@arg (list of ints) included_rounds: list of rounds between 0 and R - 1 whose stochastic events we wish to 
		include in the empirical game model
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game
	@arg (map) coarse_to_true_histories_map: Map of each coarsened infoset in empirical game to corr. infoset in true game
	@arg (dict) hp_set: map of set hyperparameters with the following keys:
		num_training_steps, gamma, epsilon_min, epsilon_annealing, learning_rate, model_width, update_target

	Compute Player 2's best response for each infoset without the true game ExtensiveForm object,
	using a DQN
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]
	num_training_steps = hp_set["training_steps"]

	trials = int(num_training_steps)
	trial_len = 500

	state_len = get_state_length(num_rounds)
	action_space = p2_actions[:]
	dqn_agent = DQN((state_len,), action_space, hp_set)
	steps = 0
	eval_over_time = []
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	for trial in range(trials):
		if trial % 1000 == 0:
			print("BR for player 2, trial # ", trial)

		empir_history = ()
		cur_history = ()
		outcome, event_index = sample_stochastic_event_given_history(cur_history, chance_events, card_weights)
		cur_history = cur_history + (outcome,)
		empir_history = empir_history + (outcome,)
		
		done = False
		empir_infoset_id1 = get_empirical_infoset_id_given_empir_history(empir_history, 1)
		infoset_strat1 = meta_strategy.get(empir_infoset_id1, None)
		round_num = math.floor(len(cur_history) / 3)

		w = []
		a_space = []
		
		for empir_action1 in infoset_strat1.keys():
			a_space.append(empir_action1)
			w.append(infoset_strat1.get(empir_action1))

		p1_empir_action = random.choices(a_space, weights=w)[0]
		empir_history = empir_history + (p1_empir_action,)
		p1_action = None

		if round_num in included_rounds:
			p1_action = p1_empir_action
		else:
			p1_action = get_action_given_policy(p1_empir_action, p1_actions, POLICY_SPACE1, cur_history, game_params)

		cur_history = cur_history + (p1_action,)
		cur_state = convert_into_state(cur_history, num_rounds, p1_actions, p2_actions, chance_events)

		for step in range(trial_len):
			steps += 1
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))
			action = dqn_agent.act(cur_state_arr, steps)
			empir_policy = None
			round_num = math.floor(len(cur_history) / 3)

			if round_num in included_rounds:
				empir_policy = action
			else:
				empir_policy = get_policy_given_action(action, p2_actions, POLICY_SPACE2, cur_history, game_params)
			
			next_state, reward, done, next_history, next_empir_history = player2_step(meta_strategy, action, empir_policy, cur_history, empir_history, cur_state, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2)
			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))
			assert len(next_state) == state_len
			dqn_agent.remember(cur_state_arr, action, reward, next_state_arr, done)

			cur_state = next_state[:]
			cur_history = next_history
			empir_history = next_empir_history
			avg_q_output = dqn_agent.replay(steps)
			
			if done:
				break

			if steps >= num_training_steps:
				break

		if steps >= num_training_steps:
			break

		'''
		if trial % 200 == 0:
			br2_pay = evaluate_player2(dqn_agent, meta_strategy, file_ID, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2)
			eval_over_time.append(br2_pay)
			# print("BR2_pay ", br2_pay, " for trial # ", trial)
		'''

	final_model_name = "success_2_" + file_ID + "_" + str(included_rounds[-1]) + "_" + mss_type + ".model"
	dqn_agent.save_model(final_model_name)
	reconstructed_model = tf.keras.models.load_model(final_model_name)
	BR2_weights = reconstructed_model.get_weights()

	del dqn_agent
	del reconstructed_model
	gc.collect()
	shutil.rmtree(final_model_name)

	if is_regret_eval:
		return BR2_weights, POLICY_SPACE2

	else:
		#return BR2_weights, eval_over_time, POLICY_SPACE2
		return BR2_weights, POLICY_SPACE2

def player1_step(meta_strategy, p1_action, p1_empir_action, cur_history, cur_empir_history, cur_state, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
	'''
	@arg (map: Infoset --> (map: str --> float)) meta_strategy: Current metastrategy. Each key in the 
		outer map is a player infoset. Each player infoset's strategy is represented as a second map 
		giving a distribution over that infoset's action space
	@arg (list of ints) included_rounds: list of rounds between 0 and R - 1 whose stochastic events we wish to 
		include in the empirical game model
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends
	@arg (str) p1_action: Chosen action of player 1 to be played out
	@arg (tup) cur_history: History corresponding to the current player 1 node in the true game
	@arg (tup) coarsened_history: History corresponding to coarsened player 1 node in the empirical game
	@arg (map) coarse_to_true_histories_map: Map of each coarsened infoset in empirical game to corr. infoset in true game
	@arg (list of ints (0/1)) cur_state: One-hot encoding of cur_history

	Steps through true game environment given the current state and player 1's chosen action; returns
	the next state, reward (if any), and updated history. Corresponds to env.step() function one might
	find when applying DQNs to a gym env
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]
	
	next_history = cur_history + (p1_action,)
	next_empir_history = cur_empir_history + (p1_empir_action,)
	next_state = cur_state[:]
	next_state = convert_into_state(next_history, num_rounds, p1_actions, p2_actions, chance_events)
	round_num = math.floor(len(next_history) / 3)
	
	reward = 0
	w = []
	a_space = []
	done = False

	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)
	empir_infoset_id2 = get_empirical_infoset_id_given_empir_history(next_empir_history, 2)
	infoset_strat2 = meta_strategy.get(empir_infoset_id2, None)

	if infoset_strat2 is None:
		if len(next_history) == 2:
			a_space = [default_policy2]
			w = [1.0]
		else:
			last_policy_str = None
			
			if round_num in included_rounds:
				last_p2_action = empir_infoset_id2[1][-2]
				last_policy_str = get_last_policy_str(last_p2_action, next_history, p2_actions, POLICY_SPACE2, game_params)
				a = get_action_given_policy(last_policy_str, p2_actions, POLICY_SPACE2, next_history, game_params)
				a_space = [a]
				w = [1.0]
			else:
				last_action = empir_infoset_id2[1][-1]
				
				if last_action in p2_actions:
					last_policy_str = get_last_policy_str(last_action, next_history, p2_actions, POLICY_SPACE2, game_params)
					
				else:
					last_policy_str = last_action

				a_space = [last_policy_str]
				w = [1.0]
			
			if last_policy_str is None:
				last_policy_str = default_policy2
				a_space = [last_policy_str]
				w = [1.0]

	else:
		for empir_action2 in infoset_strat2.keys():
			a_space.append(empir_action2)
			w.append(infoset_strat2.get(empir_action2))

	p2_empir_action = random.choices(a_space, weights=w)[0]
	next_empir_history = next_empir_history + (p2_empir_action,)
	p2_action = None

	if round_num in included_rounds:
		p2_action = p2_empir_action
	else:

		p2_action = get_action_given_policy(p2_empir_action, p2_actions, POLICY_SPACE2, next_history, game_params)
	
	next_history = next_history + (p2_action,)
	next_state = convert_into_state(next_history, num_rounds, p1_actions, p2_actions, chance_events)
	num_p2_actions = len([x for x in next_history if x in p2_actions])

	if num_p2_actions == (num_rounds - 1):
		reward = get_utility(list(next_history), num_rounds, payoff_map)[0]
		done = True
		next_state[-1] = 1
	else:
		outcome, event_index = sample_stochastic_event_given_history(next_history, chance_events, card_weights)
		if round_num + 1 in included_rounds:
			next_empir_history = next_empir_history + (outcome,)

		next_history = next_history + (outcome,)
		next_state = convert_into_state(next_history, num_rounds, p1_actions, p2_actions, chance_events)
	
	return next_state, reward, done, next_history, next_empir_history


def player2_step(meta_strategy, p2_action, p2_empir_action, cur_history, cur_empir_history, cur_state, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
	'''
	@arg (map: Infoset --> (map: str --> float)) meta_strategy: Current metastrategy. Each key in the 
		outer map is a player infoset. Each player infoset's strategy is represented as a second map 
		giving a distribution over that infoset's action space
	@arg (list of ints) included_rounds: list of rounds between 0 and R - 1 whose stochastic events we wish to 
		include in the empirical game model
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends	
	@arg (str) p2_action: Chosen action of player 2 to be played out
	@arg (tup) cur_history: History corresponding to the current player 2 node in the game
	@arg (tup) coarsened_history: History corresponding to coarsened player 2 node in the empirical game
	@arg (map) coarse_to_true_histories_map: Map of each coarsened infoset in empirical game to corr. infoset in true game
	@arg (list of ints (0/1)) cur_state: One-hot encoding of cur_history
	@arg (str) last_p1_action: Chosen action of player 1 that led to the current node; included so that it
		can be added to the next state (player 2 hasn't seen it yet)

	Steps through true game environment given the current state and player 2's chosen action; returns
	the next state, reward (if any), and updated history. Corresponds to env.step() function one might
	find when applying DQNs to a gym env
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]
	
	next_empir_history = cur_empir_history + (p2_empir_action,)
	next_history = cur_history + (p2_action,)
	next_state = convert_into_state(next_history, num_rounds, p1_actions, p2_actions, chance_events)

	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)
	p1_action = None
	num_p2_actions = len([x for x in next_history if x in p2_actions])

	if num_p2_actions == (num_rounds - 1):
		reward = get_utility(list(next_history), num_rounds, payoff_map)[1]
		done = True
		next_state[-1] = 1
	else:
		outcome, event_index = sample_stochastic_event_given_history(next_history, chance_events, card_weights)
		next_history = next_history + (outcome,)
		next_state = convert_into_state(next_history, num_rounds, p1_actions, p2_actions, chance_events)
		round_num = math.floor(len(next_history) / 3)

		if round_num in included_rounds:
			next_empir_history = next_empir_history + (outcome,)
		
		empir_infoset_id1 = get_empirical_infoset_id_given_empir_history(next_empir_history, 1)
		infoset_strat1 = meta_strategy.get(empir_infoset_id1, None)

		w = []
		a_space = []
		done = False
		reward = 0
		
		if infoset_strat1 is None:
			if len(next_history) == 1:
				a_space = [default_policy1]
				w = [1.0]
			else:
				last_policy_str = None
				if round_num in included_rounds:
					last_p1_action = empir_infoset_id1[1][-3]
					last_policy_str = get_last_policy_str(last_p1_action, next_history, p1_actions, POLICY_SPACE1, game_params)
					a = get_action_given_policy(last_policy_str, p1_actions, POLICY_SPACE1, next_history, game_params)
					a_space = [a]
					w = [1.0]
				else:
					last_action = empir_infoset_id1[1][-2]
					if last_action in p1_actions:
						last_policy_str = get_last_policy_str(last_action, next_history, p1_actions, POLICY_SPACE1, game_params)
					else:
						last_policy_str = last_action
					
					a_space = [last_policy_str]
					w = [1.0]
				
				if last_policy_str is None:
					last_policy_str = default_policy1
					a_space = [last_policy_str]
					w = [1.0]
		else:
			for empir_action1 in infoset_strat1.keys():
				a_space.append(empir_action1)
				w.append(infoset_strat1.get(empir_action1))

		p1_empir_action = random.choices(a_space, weights=w)[0]
		next_empir_history = next_empir_history + (p1_empir_action,)
		p1_action = None

		if round_num in included_rounds:
			p1_action = p1_empir_action
		else:
			p1_action = get_action_given_policy(p1_empir_action, p1_actions, POLICY_SPACE1, cur_history, game_params)
		
		next_history = next_history + (p1_action,)
		next_state = convert_into_state(next_history, num_rounds, p1_actions, p2_actions, chance_events)

	return next_state, reward, done, next_history, next_empir_history


def compute_best_response(meta_strategy, mss_type, file_ID, game_param_map, hp_set1, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, is_regret_eval):
	'''
	@arg (map: Infoset --> (map: str --> float)) meta-strategy: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: corresponding ID for the given game/DQN
	@arg (list of ints) included_rounds: list of rounds between 0 and R - 1 whose stochastic events we wish to 
		include in the empirical game model
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends
	@arg (map) coarse_to_true_histories_map: Map of each coarsened infoset in empirical game to corr. infoset in true game
	@arg (dict) hp_set: map of set hyperparameters with the following keys:
		num_training_steps, gamma, epsilon_min, epsilon_annealing, learning_rate, model_width, update_target

	Compute each player's best response for each infoset without the true game ExtensiveForm object,
	using tabular Q-learning
	'''
	#br1, br1_pay, POLICY_SPACE1 = dqn_br_player_1(meta_strategy, mss_type, file_ID, game_param_map, hp_set1, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, is_regret_eval)
	br1, POLICY_SPACE1 = dqn_br_player_1(meta_strategy, mss_type, file_ID, game_param_map, hp_set1, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, is_regret_eval)
	
	#br2, br2_pay, POLICY_SPACE2 = dqn_br_player_2(meta_strategy, mss_type, file_ID, game_param_map, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, is_regret_eval)
	br2, POLICY_SPACE2 = dqn_br_player_2(meta_strategy, mss_type, file_ID, game_param_map, hp_set2, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2, is_regret_eval)

	# return br1, br2, POLICY_SPACE1, POLICY_SPACE2, br1_pay, br2_pay
	return br1, br2, POLICY_SPACE1, POLICY_SPACE2

def evaluate_player1(dqn_agent, meta_strategy, file_ID, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
	'''
	@arg (DQN) dqn_agent: agent for DQN representing player 1
	@arg (map: tuple --> (map: str --> float)) meta-strategy: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: string corresponding to this particular network for training
	@arg (list of ints) included_rounds: list of rounds between 0 and R - 1 whose stochastic events we wish to 
		include in the empirical game model
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends
	@arg (map) coarse_to_true_histories_map: Map of each coarsened infoset in empirical game to corr. infoset in true game

	Evaluates the average reward over a series of episodes (simulated gameplay) to player 1 given the 
	now-trained DQN
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)
	
	trial_len = 500
	total_reward_over_time = 0
	
	# final bit for "done"
	state_len = get_state_length(num_rounds)
	action_space = p1_actions[:]

	for ep in range(NUM_EVAL_EPISODES):
		if ep % 100 == 0:
			print("Evaluation for player 1, episode # ", ep)

		cur_history = ()
		empir_history = ()
		outcome, event_index = sample_stochastic_event_given_history(cur_history, chance_events, card_weights)
		cur_history = cur_history + (outcome,)
		empir_history = empir_history + (outcome,)
		
		done = False
		cur_state = convert_into_state(cur_history, num_rounds, p1_actions, p2_actions, chance_events)

		for step in range(trial_len):
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))
			action = dqn_agent.act_in_eval(cur_state_arr)
			empir_policy = get_policy_given_action(action, p1_actions, POLICY_SPACE1, cur_history, game_params)
			
			next_state, reward, done, next_history, next_empir_history = player1_step(meta_strategy, action, empir_policy, cur_history, empir_history, cur_state, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2)
			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))
			cur_state = next_state[:]
			cur_history = next_history
			empir_history = next_empir_history
			
			if done:
				total_reward_over_time += reward
				break

	final = float(total_reward_over_time) / NUM_EVAL_EPISODES

	return float(total_reward_over_time) / NUM_EVAL_EPISODES


def evaluate_player2(dqn_agent, meta_strategy, file_ID, game_param_map, POLICY_SPACE1, POLICY_SPACE2, default_policy1, default_policy2):
	'''
	@arg (DQN) dqn_agent: agent for DQN representing player 2
	@arg (map: tuple --> (map: str --> float)) meta-strategy: each key
		in the outer map is a player infoset. Each player infoset's strategy is represented
		as a second map giving a distribution over that infoset's action space
	@arg (str) file_ID: string corresponding to this particular network for training
	@arg (list of ints) included_rounds: list of rounds between 0 and R - 1 whose stochastic events we wish to 
		include in the empirical game model
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends
	@arg (map) coarse_to_true_histories_map: Map of each coarsened infoset in empirical game to corr. infoset in true game

	Evaluates the average reward over a series of episodes (simulated gameplay) to player 2 given the 
	now-trained DQN
	'''
	num_rounds = game_param_map["num_rounds"]
	p1_actions = game_param_map["p1_actions"]
	p2_actions = game_param_map["p2_actions"]
	chance_events = game_param_map["chance_events"]
	card_weights = game_param_map["card_weights"]
	payoff_map = game_param_map["payoff_map"]
	included_rounds = game_param_map["included_rounds"]
	game_params = (num_rounds, included_rounds, p1_actions, p2_actions, chance_events)

	trial_len = 500
	total_reward_over_time = 0
	state_len = get_state_length(num_rounds)
	action_space = p2_actions[:]

	for ep in range(NUM_EVAL_EPISODES):
		if ep % 100 == 0:
			print("Evaluation for player 2, episode # ", ep)

		cur_history = ()
		empir_history = ()
		outcome, event_index = sample_stochastic_event_given_history(cur_history, chance_events, card_weights)
		cur_history = cur_history + (outcome,)
		empir_history = empir_history + (outcome,)
		done = False
		
		empir_infoset_id1 = get_empirical_infoset_id_given_empir_history(empir_history, 1)
		infoset_strat1 = meta_strategy.get(empir_infoset_id1, None)
		round_num = math.floor(len(cur_history) / 3)
		w = []
		a_space = []
		
		if infoset_strat1 is None:
			if len(cur_history) == 1:
				a_space = [default_policy1]
				w = [1.0]
			else:
				last_policy_str = None
				if round_num in included_rounds:
					last_policy_str = empir_infoset_id1[1][-3]
				else:
					last_policy_str = empir_infoset_id1[1][-2]
				
				if last_policy_str is None:
					last_policy_str = default_policy1

				a_space = [last_policy_str]
				w = [1.0]
		else:
			for empir_action1 in infoset_strat1.keys():
				a_space.append(empir_action1)
				w.append(infoset_strat1.get(empir_action1))

		p1_empir_action = random.choices(a_space, weights=w)[0]
		p1_action = get_action_given_policy(p1_empir_action, p1_actions, POLICY_SPACE1, cur_history, game_params)
		cur_history = cur_history + (p1_action,)
		cur_empir_history = cur_empir_history + (p1_empir_action,)

		cur_state = convert_into_state(cur_history, num_rounds, p1_actions, p2_actions, chance_events)

		for step in range(trial_len):
			x = np.array(cur_state)
			cur_state_arr = x.reshape(-1, len(cur_state))
			action = dqn_agent.act_in_eval(cur_state_arr)
			empir_policy = get_policy_given_action(action, p2_actions, POLICY_SPACE2, cur_history, game_params)
			
			next_state, reward, done, next_history, next_empir_history = player2_step(meta_strategy, action, empir_policy, cur_history, cur_state, game_param_map, POLICY_SPACE1, POLICY_SPACE2, 
				default_policy1, default_policy2)
			x = np.array(next_state)
			next_state_arr = x.reshape(-1, len(next_state))
			assert len(next_state) == state_len

			cur_state = next_state[:]
			cur_history = next_history
			empir_history = next_empir_history
			
			if done:
				total_reward_over_time += reward
				break		


	final = float(total_reward_over_time) / NUM_EVAL_EPISODES

	return float(total_reward_over_time) / NUM_EVAL_EPISODES

