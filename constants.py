"""Define constants for the Easy21 assignment."""

"""
==== System constants ====
"""
VERBOSE = False


"""
==== Easy21 environment constants ====
"""
# Game constants
CARD_VALUES = range(1, 11)
TERMINAL = 1
NON_TERMINAL = 0
LIMIT = 21
DEALER_MIN = 17
RED = 1
BLACK = 0

# Rewards
WIN = 1         # End of game, player wins.
LOSS = -1       # End of game, player loses.
DRAW = 0        # End of game, players drawn.
CONTINUE = 0    # Middle of game, continue playing.

# State and action spaces
HIT = 1
STICK = 0
ACTIONS = [STICK, HIT]
ACTION_SPACE = len(ACTIONS)
PLAYER_STATES = range(1, 22)
PLAYER_STATE_SPACE = len(PLAYER_STATES)
DEALER_STATES = CARD_VALUES
DEALER_STATE_SPACE = len(DEALER_STATES)
STATE_SPACE = (PLAYER_STATE_SPACE, DEALER_STATE_SPACE)
STATE_ACTION_SPACE = (PLAYER_STATE_SPACE, DEALER_STATE_SPACE, ACTION_SPACE)


"""
==== Monte-Carlo control agent constants ====
"""
# Learning constants
MC_NUM_EPISODES = 1000 * 500
N_0 = 100

"""
==== Sarsa control agent constants ====
"""
SARSA_NUM_EPISODES = 1000
LAMBDA = 0.9
