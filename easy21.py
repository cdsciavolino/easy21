"""
Implementation of Easy21 Assignment
Q1: Implementation of Easy21
"""
import random
import constants as const


class State:
    """Wrapper class for the current state of the game."""

    def __init__(self, dealer, player, terminal):
        self.dealer = dealer
        self.player = player
        self.terminal = terminal

    def __str__(self):
        return "{}Dealer: {} vs. Player: {}".format(
            '(T) ' if self.terminal == const.TERMINAL
            else '', self.dealer, self.player)

    def copy(self):
        """Return a copy of the current state."""
        return State(self.dealer, self.player, self.terminal)


class Card:
    """Wrapper class for card that has a color and value."""

    def __init__(self, color, value):
        self.color = color
        self.value = value

    def __str__(self):
        return "({}, {})".format(
            'R' if self.color == const.RED else 'B', self.value)


def draw_card():
    """
    Draws a card from the deck. Values are between [1,10] and the color is
    red with probability 1/3 and black with probability 2/3.
    """
    rand_value = random.choice(const.CARD_VALUES)
    rand_color = const.RED if random.randint(1, 3) == 1 else const.BLACK
    return Card(rand_color, rand_value)


def draw_first_card():
    """
    Draws the first cards of the game, which are guaranteed to be black.
    """
    return Card(const.BLACK, random.choice(const.CARD_VALUES))


def bust_hand(amt):
    """True iff the given hand value `amt` is a bust. False otherwise."""
    return amt > const.LIMIT or amt < 1


def handle_hit(state):
    """
    Private method: Given a state, simulate the hit action and return the
    associated state.
    """
    next_card = draw_card()
    printop('  > Player draws {}'.format(next_card))
    if next_card.color == const.RED:
        state.player -= next_card.value
    else:
        state.player += next_card.value
    return state


def handle_stick(state):
    """
    Private method: Given a state, simulate the stick action and return the
    associated state.
    """
    # Dealer draws cards until it hits DEALER_MIN or busts
    while not bust_hand(state.dealer) and state.dealer < const.DEALER_MIN:
        next_card = draw_card()
        printop('  > Dealer draws {}'.format(next_card))
        if next_card.color == const.RED:
            state.dealer -= next_card.value
        else:
            state.dealer += next_card.value
    return state


def step(state, action):
    """
    Given a state (dealer,player) cards and action (HIT or STICK), will
    return the next state and the associated reward in that order
    """
    state = state.copy()
    next_state = None
    printop('  > Player {}!'.format(
        'hits' if action == const.HIT else 'sticks'))
    if action == const.HIT:
        next_state = handle_hit(state)
    if action == const.STICK:
        next_state = handle_stick(state)

    # Player bust -> Loss
    if bust_hand(next_state.player):
        printop('  > Player busts -> Loss!')
        next_state.terminal = const.TERMINAL
        return next_state, const.LOSS

    # Dealer bust -> Win
    if bust_hand(next_state.dealer):
        printop('  > Dealer busts -> Win!')
        next_state.terminal = const.TERMINAL
        return next_state, const.WIN

    # Determine winner on player stick action
    if action == const.STICK:
        next_state.terminal = const.TERMINAL
        if next_state.player == next_state.dealer:
            printop('  > Tied scores -> Draw!')
            return next_state, const.DRAW
        if next_state.player > next_state.dealer:
            printop('  > Player has higher score -> Win!')
            return next_state, const.WIN
        printop('  > Dealer has higher score -> Loss!')
        return next_state, const.LOSS

    # Return the (next_state, reward) tuple
    return next_state, const.CONTINUE


def new_game():
    """Return an initial state with cards drawn for the dealer and player."""
    return State(
        random.choice(const.CARD_VALUES),
        random.choice(const.CARD_VALUES),
        const.NON_TERMINAL
    )


def printop(string):
    """Optionally prints the string if const.VERBOSE is True."""
    if const.VERBOSE:
        print(string)


if __name__ == '__main__':
    game = new_game()
    print(game)
    game, rwd = step(game, const.HIT)
    print("Reward: {}".format(rwd))
    print(game)
    game, rwd = step(game, const.STICK)
    print("Reward: {}".format(rwd))
    print(game)
