from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
from ExtensiveForm import *
from Infoset import *
from Node import *
try:
    from reprlib import repr
except ImportError:
    pass

import gc

gc.enable()

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o)) #file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s


    # size in Bytes
    # return sizeof(o)

    # size in MB
    # return sizeof(o) * 1.0e-6

    # size in GB
    return sizeof(o) * 1.0e-9

def ExtensiveFormHandler(obj):
    '''
    '''
    assert isinstance(obj, ExtensiveForm)
    yield obj.infosets
    yield obj.terminal_nodes
    yield obj.root
    yield obj.chance_map
    yield obj.num_players
    yield obj.num_rounds

def InfosetHandler(obj):
    '''
    '''
    assert isinstance(obj, Infoset)
    yield obj.infoset_id
    yield obj.node_list
    yield obj.action_space
    yield obj.strategy
    yield obj.regret_sum
    yield obj.strategy_sum
    yield obj.reach_prob_sum
    yield obj.reach_prob
    yield obj.action_utils
    yield obj.num_players


def NodeHandler(obj):
    '''
    '''
    assert isinstance(obj, Node)
    yield obj.num_players
    yield obj.player_id
    yield obj.infoset_id
    yield obj.history
    yield obj.children
    yield obj.strategy
    yield obj.is_terminal
    yield obj.is_chance
    yield obj.utility
    yield obj.action_space


# prefix = "test_job_output_4_1/T500_BIG_DoND_161GZ_NE_mss_SPE_eval"

# POLICY_SPACE1 = np.load(prefix + "_policy_map1.npy", allow_pickle=True).item()
# POLICY_SPACE2 = np.load(prefix + "_policy_map2.npy", allow_pickle=True).item()
# print(total_size(POLICY_SPACE1))
# print(total_size(POLICY_SPACE2))

##### Example call #####

# if __name__ == '__main__':
#     d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
#     print(total_size(d, verbose=True))