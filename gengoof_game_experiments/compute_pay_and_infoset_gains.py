from best_response import *

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
		pay = np.zeros(2)
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
		pay = np.zeros(2)
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
		pay = np.zeros(2)
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
		pay = np.zeros(2)
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

