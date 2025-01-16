import random
import string
import numpy as np
import math
import itertools as it

def get_utility(history, num_rounds, payoff_map):
	'''
	@arg (list) history: History at each round of our big general-sum abstract game at the
		current terminal node being implemented. Can be divided into groups of 3 for each round:
		(event, player 1's move, player 2's move)
	@arg (map) payoff_map: Map of each (card, p1_action, p2_action) to a unique utility for that
		particular round in the true game; sum of these across all rounds corresponds to the payoff
		at the leaf once the game ends
	
	Helper method for computing the final utility at the end of a single playthrough
	'''
	final_u = np.zeros(2)
	for r in range(num_rounds - 1):
		hist_r = history[3 * r : 3 * (r + 1)]
		final_u += payoff_map.get(tuple(hist_r))

	return final_u

def get_chance_node_dist_given_history(history, chance_events, card_weights):
	'''
	@arg (list) history: History at each round of our big general-sum abstract game at the
		current terminal node being implemented. Can be divided into groups of 3 for each round:
		(event, player 1's move, player 2's move)
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game

	Returns the corresponding distribution of events ("cards") for the chance node represented
	by the input history, given the weights of all cards for the game
	'''
	history_events = [e for e in history if e in chance_events]
	available_events = [e for e in chance_events if e not in history_events]
	prob_dist = {}
	denom = sum([card_weights[card] for card in available_events])

	for card in available_events:
		prob_dist[card] = card_weights[card] / float(denom)

	return prob_dist

def sample_stochastic_event_given_history(history, chance_events, card_weights):
	'''
	@arg (list) history: History at each round of our big general-sum abstract game at the
		current terminal node being implemented. Can be divided into groups of 3 for each round:
		(event, player 1's move, player 2's move)
	@arg (map) card_weights: Map of each card to its corresponding weight for the given game;
		distribution is randomly generated for each game

	Samples an event for the given chance node in the game
	'''
	history_events = [e for e in history if e in chance_events]
	available_events = [e for e in chance_events if e not in history_events]
	prob_dist = get_chance_node_dist_given_history(history, chance_events, card_weights)
	rand_num = random.random()
	cum_prob = 0.0
	for x in prob_dist.keys():
		cum_prob += prob_dist[x]

		if rand_num < cum_prob:
			return x, available_events.index(x)

def get_coarsened_infoset(coarsened_history):
	'''
	@arg (tuple) coarsened_history: history of player actions and stochastic outcomes,
		as they appear according to the coarseness of the empirical game model

	Helper function for computing the coarsened infoset number; designed to replace
	the true_to_coarsened_infosets_map object and save memory
	'''
	coarse_id = (1, coarsened_history)
	num_player_moves = len(coarsened_history) - len(included_stochastic_rounds)

	if num_player_moves % 2 == 1:
		coarse_id = (2, coarsened_history[:-1])

	return coarse_id



