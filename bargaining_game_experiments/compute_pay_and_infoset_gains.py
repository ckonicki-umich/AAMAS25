from bargaining import *
from best_response import *

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
		pay = np.zeros(2)
		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, v1, None, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
		
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		pay = np.zeros(2)	
		for o1 in OUTSIDE_OFFERS:
			prob = outside_offer_dist1.get(o1)
			next_node = action_history + (o1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_pay_helper(next_node, strategy_profile, BR1_weights, BR2_weights, br_player, 
				next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, v1, None, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# third chance node has been reached -- valuation for player 2 in true game
	elif len(action_history) == 2:
		pay = np.zeros(2)
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
		pay = np.zeros(2)
		for o2 in OUTSIDE_OFFERS:
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
		pay = np.zeros(2)
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

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(action_history) == 0:
		pay = np.zeros(2)

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, None, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
		
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(action_history) == 1:
		pay = np.zeros(2)	
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
		pay = np.zeros(2)
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = action_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(action_history) == 3:
		pay = np.zeros(2)
		for o2 in OUTSIDE_OFFERS:
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
		pay = np.zeros(2)
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
				if last_policy_str[0] is None:
					last_policy_str = "pi_0"

				policy_str = last_policy_str

			empir_action = (policy_str, signal)
			offer = get_offer_given_policy(policy_str, action_space, PS, action_history, pool)
			a = (offer, signal)
			next_node = action_history + (a,)
			next_reach_prob = input_reach_prob
			new_pay, POLICY_SPACE1, POLICY_SPACE2 = recursive_true_empirical_strategy_pay_helper(next_node, strategy_profile, next_reach_prob, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
				o1_pay_arr, o2_pay_arr, v1, v2, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

	return pay, POLICY_SPACE1, POLICY_SPACE2

def compute_empirical_pay_given_infoset(br_meta_strat, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	return recursive_empirical_pay_infoset_helper(br_meta_strat, [], [], 1.0, 1.0, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
		o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)


def recursive_empirical_pay_infoset_helper(br_meta_strat, true_history, empir_history, input_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	turns_taken = len(true_history) - 4
	num_prev_rounds = turns_taken // 2
	infoset_freq = None

	if len(true_history) == 0:
		pay = 0.0
		infoset_freq = 0.0

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
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
			next_node = true_history + (o1,)
			next_empir_history = (o1,)
			next_reach_prob = input_reach_prob
			next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist1.get(o1)
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			infoset_freq = new_infoset_freq

		else:
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			# this is player 2's infoset
			if signal_count == 2:
				o1 = infoset_id[1][0]
				next_node = true_history + (o1,)
				next_empir_history = (o1,)
				next_reach_prob = input_reach_prob
				next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist1.get(o1)
				new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				infoset_freq = new_infoset_freq

			else:
				for o1 in OUTSIDE_OFFERS:
					prob = outside_offer_dist1.get(o1)
					next_node = true_history + (o1,)
					next_empir_history = (o1,)
					next_reach_prob = input_reach_prob * prob
					next_infoset_reach_prob = infoset_reach_prob * prob
					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					infoset_freq = infoset_freq + new_infoset_freq

	# third chance node has been reached -- valuation for player 2 in true game
	elif len(true_history) == 2:
		pay = 0.0
		infoset_freq = 0.0
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = true_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			infoset_freq = new_infoset_freq

	# fourth chance node has been reached -- outside offer for player 2 in true game
	elif len(true_history) == 3:
		pay = 0.0
		infoset_freq = 0.0
		br_player = infoset_id[0]
		if br_player == 1:
			# this infoset belongs to player 1
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			if signal_count == 2:
				o2 = infoset_id[1][1]
				next_node = true_history + (o2,)
				next_empir_history = empir_history + (o2,)
				next_reach_prob = input_reach_prob
				next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist2.get(o2)
				new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				infoset_freq = new_infoset_freq
			else:
				for o2 in OUTSIDE_OFFERS:
					prob = outside_offer_dist2.get(o2)
					next_node = true_history + (o2,)
					next_empir_history = empir_history + (o2,)
					next_reach_prob = input_reach_prob * prob
					next_infoset_reach_prob = infoset_reach_prob * prob
					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					infoset_freq = infoset_freq + new_infoset_freq

		else:
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			o2 = None
			if signal_count == 2:
				o2 = infoset_id[1][1]
			else:
				o2 = infoset_id[1][0]
			
			next_node = true_history + (o2,)
			next_empir_history = empir_history + (o2,)
			next_reach_prob = input_reach_prob
			next_infoset_reach_prob = infoset_reach_prob * outside_offer_dist2.get(o2)
			new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
			infoset_freq = new_infoset_freq

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

		if player_num != br_player:
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, br_player)
			empir_infoset_id = None
			if offer_reveal:
				empir_infoset_id = (player_num, empir_history)
			else:
				if br_player == 1:
					empir_infoset_id = (player_num, empir_history[1:])
				else:
					empir_infoset_id = (player_num, (empir_history[0],) + empir_history[2:])

			assert empir_infoset_id[0] == player_num
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			
			if len(empir_actions) < len(input_empir_actions):
				# choose other player's actions w/ prob 1 so that they lead to the input br_player's infoset
				infoset_freq = 0.0
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				next_empir_history = empir_history + (empir_action,)
				infoset_strat = br_meta_strat.get(empir_infoset_id)
				
				if infoset_strat is not None:
					prob = infoset_strat.get(empir_action, 0.0)
					if prob > 0.0:
						next_infoset_reach_prob = prob * infoset_reach_prob
						new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
							outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
						pay = pay + new_pay
						infoset_freq = infoset_freq + new_infoset_freq

			else:
				# follow br_meta_strat after input br_player's infoset
				infoset_strat = br_meta_strat.get(empir_infoset_id)
				game_start = len(true_history) <= 4

				if infoset_strat is not None:
					infoset_freq = 0.0
					for empir_action in infoset_strat.keys():
						policy_str = empir_action[0]
						if infoset_strat.get(empir_action, 0.0) > 0.0:
							offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
							next_node = true_history + ((offer, empir_action[1]),)
							next_empir_history = empir_history + (empir_action,)
							next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)
							new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
								outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
							pay = pay + new_pay
							infoset_freq = new_infoset_freq

				else:
					infoset_freq = 0.0
					signal = random.choice([True, False])
					
					if len(true_history) <= 5:
						policy_str = "pi_0"
					else:
						last_policy_str = empir_infoset_id[1][-2][0]
						if last_policy_str[0] is None:
							last_policy_str = "pi_0"

						policy_str = last_policy_str

					empir_action = (policy_str, signal)
					offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
					a = (offer, signal)
					next_node = true_history + (a,)
					next_empir_history = empir_history + (empir_action,)
					next_reach_prob = input_reach_prob

					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					infoset_freq = new_infoset_freq

		else:
			other_player = player_num % 2 + 1
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, other_player)
			empir_infoset_id = None

			if offer_reveal:
				empir_infoset_id = (br_player, empir_history)
			else:
				if other_player == 1:
					empir_infoset_id = (br_player, empir_history[1:])
				else:
					empir_infoset_id = (br_player, (empir_history[0],) + empir_history[2:])
			
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			
			# choose br_player's actions so they lead to given infoset
			if len(empir_actions) < len(input_empir_actions):
				infoset_freq = 0.0
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				next_empir_history = empir_history + (empir_action,)
				infoset_strat = br_meta_strat.get(empir_infoset_id)
				prob = infoset_strat.get(empir_action, 0.0)
				if prob > 0.0:
					next_infoset_reach_prob = infoset_reach_prob * prob
					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, next_infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					infoset_freq = infoset_freq + new_infoset_freq

			else:
				infoset_strat = br_meta_strat.get(empir_infoset_id)
				game_start = len(true_history) <= 4

				if infoset_strat is not None:
					infoset_freq = 0.0
					for empir_action in infoset_strat.keys():
						policy_str = empir_action[0]

						if infoset_strat.get(empir_action, 0.0) > 0.0:
							offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
							next_node = true_history + ((offer, empir_action[1]),)
							next_empir_history = empir_history + (empir_action,)
							next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)
							new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
								outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
							pay = pay + new_pay
							infoset_freq = new_infoset_freq

				else:
					infoset_freq = 0.0
					signal = random.choice([True, False])
					if len(true_history) <= 5:
						policy_str = "pi_0"
					else:
						last_policy_str = empir_infoset_id[1][-2][0]
						if last_policy_str[0] is None:
							last_policy_str = "pi_0"

						policy_str = last_policy_str

					empir_action = (policy_str, signal)
					offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
					a = (offer, signal)
					next_node = true_history + (a,)
					next_empir_history =empir_history + (empir_action,)
					next_reach_prob = input_reach_prob

					new_pay, new_infoset_freq = recursive_empirical_pay_infoset_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_reach_prob, infoset_id, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay
					infoset_freq = new_infoset_freq

	return pay, infoset_freq

def compute_infoset_gain_of_best_response(br_meta_strat, infoset_id, br_player, BR_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2, 
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	empirical_pay, infoset_freq = compute_empirical_pay_given_infoset(br_meta_strat, infoset_id, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
	br_pay = recursive_infoset_gain_helper(br_meta_strat, [], [], 1.0, infoset_id, br_player, BR_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
	gain = br_pay - empirical_pay

	return gain, infoset_freq

def recursive_infoset_gain_helper(br_meta_strat, true_history, empir_history, input_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, outside_offer_dist1, outside_offer_dist2,
	o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2):
	'''
	'''
	turns_taken = len(true_history) - 4
	num_prev_rounds = turns_taken // 2

	# chance (root) node has been reached -- valuation for player 1 in true game
	if len(true_history) == 0:
		pay = 0.0

		for v1 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = (v1,)
			next_reach_prob = input_reach_prob * prob
			new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay
	
	# second chance node has been reached -- outside offer for player 1 in true game
	elif len(true_history) == 1:
		pay = 0.0
		if br_player == 1:
			o1 = infoset_id[1][0]
			next_node = true_history + (o1,)
			next_empir_history = (o1,)
			next_reach_prob = input_reach_prob
			new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
				outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
			pay = pay + new_pay

		else:
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])

			# this is player 2's infoset
			if signal_count == 2:
				o1 = infoset_id[1][0]
				next_node = true_history + (o1,)
				next_empir_history = (o1,)
				next_reach_prob = input_reach_prob
				new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
			else:

				for o1 in OUTSIDE_OFFERS:
					prob = outside_offer_dist1.get(o1)
					next_node = true_history + (o1,)
					next_empir_history = (o1,)
					next_reach_prob = input_reach_prob * prob
					new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

	# third chance node has been reached -- valuation for player 2 in true game
	elif len(true_history) == 2:
		pay = 0.0
		for v2 in val_dist.keys():
			prob = 1.0 / len(val_dist)
			next_node = true_history + (v2,)
			next_reach_prob = input_reach_prob * prob
			new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
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
				next_node = true_history + (o2,)
				next_empir_history = empir_history + (o2,)
				next_reach_prob = input_reach_prob
				new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
			else:
				for o2 in OUTSIDE_OFFERS:
					prob = outside_offer_dist2.get(o2)
					next_node = true_history + (o2,)
					next_empir_history = empir_history + (o2,)
					next_reach_prob = input_reach_prob * prob
					new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

		else:
			signal_count = len([x for x in infoset_id[1] if x in OUTSIDE_OFFERS])
			o2 = None
			if signal_count == 2:
				o2 = infoset_id[1][1]
			else:
				o2 = infoset_id[1][0]
			
			next_node = true_history + (o2,)
			next_empir_history = empir_history + (o2,)
			next_reach_prob = input_reach_prob
			new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
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

		if player_num != br_player:
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, br_player)
			empir_infoset_id = None
			if offer_reveal:
				empir_infoset_id = (player_num, empir_history)

			else:
				if br_player == 1:
					empir_infoset_id = (player_num, empir_history[1:])
				else:
					empir_infoset_id = (player_num, (empir_history[0],) + empir_history[2:])

			assert empir_infoset_id[0] == player_num
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			if len(empir_actions) <= len(input_empir_actions):
				# choose other player's actions w/ prob 1 so that they lead to the input br_player's infoset
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				next_empir_history = empir_history + (empir_action,)
				new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
			else:
				# follow br_meta_strat after input br_player's infoset
				infoset_strat = br_meta_strat.get(empir_infoset_id)
				game_start = len(true_history) <= 4

				if infoset_strat is not None:
					for empir_action in infoset_strat.keys():
						policy_str = empir_action[0]
						if infoset_strat.get(empir_action) > 0.0:
							offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
							next_node = true_history + ((offer, empir_action[1]),)
							next_empir_history = empir_history + (empir_action,)
							next_reach_prob = input_reach_prob * infoset_strat.get(empir_action, 0.0)

							new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
								outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
							pay = pay + new_pay

				else:
					signal = random.choice([True, False])
					if len(true_history) <= 5:
						policy_str = "pi_0"
					else:
						last_policy_str = empir_infoset_id[1][-2][0]
						if last_policy_str[0] is None:
							last_policy_str = "pi_0"

						policy_str = last_policy_str

					empir_action = (policy_str, signal)
					offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
					a = (offer, signal)
					next_node = true_history + (a,)
					next_empir_history =empir_history + (empir_action,)
					next_reach_prob = input_reach_prob

					new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
						outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
					pay = pay + new_pay

		else:
			other_player = player_num % 2 + 1
			offer_reveal = check_empirical_outside_offer_reveal(empir_history, other_player)
			empir_infoset_id = None
			if offer_reveal:
				empir_infoset_id = (br_player, empir_history)
			else:
				if other_player == 1:
					empir_infoset_id = (br_player, empir_history[1:])
				else:
					empir_infoset_id = (br_player, (empir_history[0],) + empir_history[2:])
			
			empir_actions = [a for a in empir_infoset_id[1] if a not in OUTSIDE_OFFERS]
			input_empir_actions = [a for a in infoset_id[1] if a not in OUTSIDE_OFFERS]
			
			# choose br_player's actions so they lead to given infoset
			if len(empir_actions) < len(input_empir_actions):
				action_index = len(empir_actions)
				empir_action = input_empir_actions[action_index]
				policy_str = empir_action[0]
				offer = get_offer_given_policy(policy_str, action_space, PS, true_history, pool)
				next_reach_prob = input_reach_prob
				next_node = true_history + ((offer, empir_action[1]),)
				next_empir_history = empir_history + (empir_action,)
				new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist,
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay

			else:
				state = convert_into_state(true_history, pool)
				best_action = get_best_action(state, BR_weights, action_space, len(true_history) <= 4)
				next_node = true_history + (best_action,)
				next_reach_prob = input_reach_prob
				policy_str = "pi_" + str(len(PS) - 1)
				next_empir_history = empir_history + ((policy_str, best_action[1]),)
				new_pay = recursive_infoset_gain_helper(br_meta_strat, next_node, next_empir_history, next_reach_prob, infoset_id, br_player, BR_weights, pool, val_dist, 
					outside_offer_dist1, outside_offer_dist2, o1_pay_arr, o2_pay_arr, POLICY_SPACE1, POLICY_SPACE2)
				pay = pay + new_pay
				
	return pay

