import numpy as np
import itertools as it
import math
import random

NUM_PLAYER_TURNS = 5
NUM_ITEM_TYPES = 3
MAX_POOL_SIZE = 7
MIN_POOL_SIZE = 5
VAL_TOTAL = 10
OUTSIDE_OFFERS = ["H", "L"]
DISCOUNT_FACTOR = 0.99

def generate_pool():
	'''
	'''
	valid = False
	while not valid:
		pool = []

		for i in range(NUM_ITEM_TYPES):
			pool.append(np.random.randint(0, MAX_POOL_SIZE))
		if sum(pool) >= MIN_POOL_SIZE and sum(pool) <= MAX_POOL_SIZE:
			valid = True

	return tuple(pool)

def generate_valuation_distribution(p):
	'''
	@arg (list) p: pool of items listed by quantity (books, hats, balls)

	Generates a probability distribution over possible agent valuations given
	a pool of items. We require (1) that the agent believes at least one item
	has positive value and (2) that v \dot p = VAL_TOTAL

	TODO 8/17: Determine with Mike whether or not we want to stick with a uniform
	distribution for these chance nodes that output v1, v2, or if we want to
	add some randomness and make the distributions more varied
	'''
	dist = {}
	total = 0
	for v_books in range(0, VAL_TOTAL + 1):
		total += v_books
		for v_hats in range(0, VAL_TOTAL + 1):
			total += v_hats
			for v_balls in range(0, VAL_TOTAL + 1):
				total += v_balls

				v = (v_books, v_hats, v_balls)
				is_nonzero = any([x > 0 for x in v])
				is_total_correct = np.dot(v, p) == VAL_TOTAL
				if is_nonzero and is_total_correct:
					if v not in dist:
						dist[v] = 1
					else:
						cur = dist[v]
						dist[v] = cur + 1

	return dist

def coarsen_valuation_distribution(val_dist):
	'''
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool

	Generates a coarser distribution over the keys of val_dist, based on how similar the values are to
	each other
	'''
	var_sum = 0.0

	VAR_TABLE = {}
	coarse_dist = {}
	true_to_coarsened_vals_map = {}
	num_vals = 0
	for k in val_dist:
		num_vals += 1
		var = np.var(k)
		var_sum += var
		VAR_TABLE[k] = var

	avg = math.sqrt(var_sum / num_vals)
	num_div = 0
	for k in val_dist:
		if math.sqrt(VAR_TABLE[k]) >= avg:
			true_to_coarsened_vals_map[k] = ("DIV",)
			num_div += 1
		else:
			true_to_coarsened_vals_map[k] = ("SIM",)

	coarse_dist[("DIV",)] = float(num_div) / num_vals
	coarse_dist[("SIM",)] = float(num_vals - num_div) / num_vals

	return coarse_dist, true_to_coarsened_vals_map


def generate_player_valuations(val_dist):
	'''
	@arg (dict) val_dist: Dictionary representing a distribution over possible player valuations given the
		item pool

	Samples a valuation for each player given the distribution over possible valuations
	'''
	v1, v2 = random.choices(list(val_dist.keys()), k=2)

	return v1, v2

def generate_offers(p, item_index):
	'''
	@arg (list) p: pool of items listed by quantity (books, hats, balls)
	@arg (str) item_index: identifies the item for which offers are generated

	Generates a list of possible partitions of the pool for a given item between
	the two players
	'''
	space = []
	for i in range(p[item_index] + 1):
		space.append((i, p[item_index] - i))

	return space

def generate_offer_space(pool):
	'''
	@arg (list) p: pool of items listed by quantity (books, hats, balls)

	Generates the space of possible offers, including walking away or accepting
	the other player's offer, given a pool of items; also indicates if they signal
	whether an outside offer was made or not
	'''
	action_spaces = []
	for item in range(3):
		action_spaces.append(generate_offers(pool, item)[:])

	offer_space = list(it.product(*action_spaces))
	offer_space += [("deal",), ("walk",)]

	return offer_space

def compute_utility(is_deal, p, v1, v2, split, o1_pay, o2_pay, num_rounds):
	'''
	@arg (tuple) is_deal: object indicating whether a player walked ("walk",) or
		agreed to a deal ("deal",) in response to the given offer denoted by split
	@arg (list) p: pool of items listed by quantity (books, hats, balls)
	@arg (tuple of int's) v1: player 1's valuation for each item in the pool
	@arg (tuple of int's) v2: player 2's valuation for each item in the pool
	@arg (tuple) split: partition of the item pool offered by the agent in 
		the format of (player1_share, player2_share) per item
	@arg (int) o1_pay: payoff to player 1 for accepting its private outside offer
	@arg (int) o2_pay: payoff to player 2 for accepting its private outside offer
	@arg (int) num_rounds: number of negotiation rounds (p1 then p2) that have elapsed
		so far -- used to incorporate discount factor into payoffs

	Computes the utility to each player for either walking altogether or for agreeing
	to a given partition of the item pool given their respective private valuations for
	the items
	'''
	pay1 = None
	pay2 = None
	if is_deal == ("walk",) or type(split) is str:
		pay1 = o1_pay * DISCOUNT_FACTOR**num_rounds
		pay2 = o2_pay * DISCOUNT_FACTOR**num_rounds

	elif is_deal == ("deal",):
		p1 = [x[0] for x in split]
		p2 = [x[1] for x in split]

		pay1 = np.dot(p1, v1) * DISCOUNT_FACTOR**num_rounds
		pay2 = np.dot(p2, v2) * DISCOUNT_FACTOR**num_rounds
		
	return np.array([pay1, pay2])

def is_offer_legal(pool, offer):
	'''
	@arg (tuple) pool: pool of items listed by quantity (books, hats, balls)
	@arg (tuple) offer: partition of the item pool offered by the agent in 
		the format of (player1_share, player2_share) per item

	An offer is legal if it's a proper subset of the current pool
	'''
	for i in range(NUM_ITEM_TYPES):
		if pool[i] != (offer[i][0] + offer[i][1]):
			return False

	return True

def generate_player_outside_offers(dist1, dist2):
	'''
	@arg (dict) dist1: true probability distribution for player 1's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")
	@arg (dict) dist2: true probability distribution for player 2's likelihood of
		receiving an attractive outside offer ("H") or a subpar outside offer ("L")

	Samples an outside offer signal for each player using the uniform distribution
	"L" signals an outside offer that isn't so attractive to the player while "H"
	signals an outside offer that is (i.e. pays well if accepted)
	'''
	sample1, sample2 = np.random.uniform(size=2)

	o1 = "H"
	if sample1 >= dist1["H"]:
		o1 = "L"

	o2 = "H"
	if sample2 >= dist2["H"]:
		o2 = "L"
	
	return o1, o2

def check_empirical_outside_offer_reveal(history, player_num):
	'''
	'''
	if len(history) <= 2:
		return False
	tuple_actions_only = [x for x in history if type(x) is tuple]
	
	for i in range(player_num - 1, len(tuple_actions_only), 2):	
		action = tuple_actions_only[i]
		if action == ('deal',) or action == ('walk',):
			return False

		is_bool = type(action[1]) is bool or type(action[1]) is np.bool_
		if is_bool and bool(action[1]):
			return True
	
	return False


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

		is_bool = type(action[1]) is bool or type(action[1]) is np.bool_
		if is_bool and bool(action[1]):
			return True
	
	return False

def generate_outside_offer_pay(o):
	'''
	@arg (str) o: private outside offer provided to one player, represented as
		a signal
	@arg (tup) v: player's private valuation for each item type

	Generate a random payoff corresponding to each outside offer signal, intended
	to be within the range of the best/worst valuations (0 to 10)
	"H" --> outside offer that yields a high payoff
	"L" --> outside offer that yields a low payoff
	'''
	min_pay = None
	max_pay = None
	if o == "H":
		min_pay = int(0.5 * VAL_TOTAL) + 1
		max_pay = VAL_TOTAL - 1
	else:
		min_pay = 1
		max_pay = int(0.5 * VAL_TOTAL) - 1
		
	return random.randint(min_pay, max_pay)

def get_pay_given_outside_offer(pay_arr, offer):
	'''
	@arg (tup) pay_arr: tuple of payoffs to agent given outside offer "H" or "L",
		respectively
	@arg (str) offer: private outside offer provided to given player, represented as
		a string signal
	'''
	ind = OUTSIDE_OFFERS.index(offer)
	return pay_arr[ind]

