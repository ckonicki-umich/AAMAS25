import gambit
from operator import mul
from functools import reduce
import numpy as np
import itertools as it
from fractions import Fraction

def write_nfg_file(U_NF, es_1, es_2, file_ID):
	'''
	Constructs a serialized string version of the normal-form empirical game
	and writes the output into an .nfg file for Gambit to process into a game object
	'''
	game_file = open("empirical_game_" + file_ID + ".nfg", "w")
	game_file.write("NFG 1 R \"Normal-form empirical game\"\n")

	game_file.write("{ \"Player 1\" \"Player 2\" }\n{ ")
	strat1_string = "{ "
	for s1 in es_1:
		s1_index = es_1.index(s1)
		strat1_string += "\"" + str(s1_index) + "\" "
	strat1_string += "} "
	game_file.write(strat1_string)

	strat2_string = "{ "
	for i in range(len(es_2)):
		s2 = es_2[i]
		strat2_string += "\"" + str(i) + "\" "
	strat2_string += "} }\n\"\" \n"
	game_file.write(strat2_string)

	pay_string = ""

	for s2 in es_2:
		for s1 in es_1:
			s = (s1, s2)

			pay = U_NF.get(s)

			pay_string += str(pay[0]) + " " + str(pay[1]) + " "
	game_file.write(pay_string[:-1] + "\n")


