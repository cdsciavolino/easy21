"""
Implements the following agents to solve Easy21.
    1.) Monte-Carlo Control
    2.) Sarsa(lambda) Control
"""
from mpl_toolkits.mplot3d import Axes3D  # Ability to plot in 3D
import easy21
import constants as const
import matplotlib.pyplot as plt
import numpy as np


def epsilon_greedy(epsilon, q, state):
    """Return an epsilon-greedy action for the given Q function."""
    if np.random.rand() < epsilon:
        return np.argmax(q[state.player-1, state.dealer-1, :])
    return np.random.choice(const.ACTION_SPACE)


class SarsaControl:
    """Agent using backward-view Sarsa(lambda) control to solve Easy21."""

    def __init__(self):
        self.state_action_counts = np.zeros(const.STATE_ACTION_SPACE)
        self.state_visits = np.zeros(const.STATE_SPACE)
        self.Q = np.zeros(const.STATE_ACTION_SPACE)

    def num_visits(self, state):
        """Return the number of times the given state has been visited."""
        return self.state_visits[state.player - 1, state.dealer - 1]

    def state_action_count(self, state, action):
        """Return the state-action count for the given state-action pair."""
        return self.state_action_counts[
            state.player - 1, state.dealer - 1, action]

    def train(self):
        """Train the agent on NUM_EPISODES to learn the action-value function"""
        elapsed_episodes = 0
        while elapsed_episodes < const.SARSA_NUM_EPISODES:
            eligibility_trace = np.zeros(const.STATE_ACTION_SPACE)
            state = easy21.new_game()
            epsilon = const.N_0 / (const.N_0 + self.num_visits(state))
            action = epsilon_greedy(epsilon, self.Q, state)
            while state.terminal != const.TERMINAL:
                # Take 1 step forward from the current state and simulate action
                next_state, reward = easy21.step(state, action)
                epsilon = const.N_0 / (const.N_0 + self.num_visits(next_state))
                alpha = 1 / self.state_action_count(state, action)
                next_action = epsilon_greedy(epsilon, self.Q, next_state)

                # Compute TD-error and increment counts
                cur_idx = (state.player, state.dealer, action)
                next_idx = (next_state.player, next_state.dealer, next_action)
                td_error = reward + self.Q[next_idx] - self.Q[cur_idx]
                eligibility_trace[cur_idx] += 1
                self.state_action_counts[cur_idx] += 1
                self.state_visits[state.player - 1, state.dealer - 1] += 1

                # Update action-value function and eligibility traces
                self.Q += alpha * td_error * eligibility_trace
                eligibility_trace *= const.LAMBDA
                state = next_state
                action = next_action

            elapsed_episodes += 1


class MonteCarloControl:
    """Agent that uses Monte-Carlo control to solve Easy21."""

    def __init__(self):
        self.state_action_counts = np.zeros(const.STATE_ACTION_SPACE)
        self.state_visits = np.zeros(const.STATE_SPACE)
        self.Q = np.zeros(const.STATE_ACTION_SPACE)

    def num_visits(self, state):
        """Return the number of times the given state has been visited."""
        return self.state_visits[state.player - 1, state.dealer - 1]

    def increment_state_visits(self, state):
        """Increments the state_visits list field."""
        self.state_visits[state.player - 1, state.dealer - 1] += 1

    def state_action_count(self, state, action):
        """Return the state-action count for the given state-action pair."""
        return self.state_action_counts[state.player-1, state.dealer-1, action]

    def increment_state_action_counts(self, state, action):
        """Increments the state_action_counts field."""
        self.state_action_counts[state.player-1, state.dealer-1, action] += 1

    def update_q_function(self, state, action, alpha, total_return):
        """Perform an update on the Q-function with the alpha, total_return."""
        cur_action_value = self.Q[state.player - 1, state.dealer - 1, action]
        self.Q[state.player - 1, state.dealer - 1, action] += (
                alpha * (total_return - cur_action_value)
        )

    def train(self):
        """Train the action-value function on NUM_EPISODES games."""
        elapsed_episodes = 0
        while elapsed_episodes < const.MC_NUM_EPISODES:
            state = easy21.new_game()
            episode = []  # [ (state_0, action_0, reward_1, state_1), ... ]

            # Experience 1 episode
            while state.terminal == const.NON_TERMINAL:
                epsilon = const.N_0 / (const.N_0 + self.num_visits(state))
                action = epsilon_greedy(epsilon, self.Q, state)
                next_state, rwd = easy21.step(state, action)
                episode.append((state, action, rwd, next_state))
                state = next_state

            # Learn from experienced episode
            total_return = episode[-1][2]
            for (state, action, reward, next_state) in episode:
                self.increment_state_action_counts(state, action)
                self.increment_state_visits(state)
                alpha_t = 1 / self.state_action_count(state, action)
                self.update_q_function(state, action, alpha_t, total_return)

            elapsed_episodes += 1

    def plot_optimal_value_function(self):
        """Plots the optimal value function on a 3d plot."""
        axes = plt.gca(projection='3d')
        v_star = np.max(self.Q, axis=2)

        X, Y = np.mgrid[const.PLAYER_STATES, const.DEALER_STATES]
        axes.plot_surface(X, Y, v_star, rstride=1, cstride=1, cmap='viridis',
                          linewidth=0, antialiased=False)
        axes.contour3D(X, Y, np.max(self.Q, axis=2))
        axes.set_xlabel('X = Player')
        axes.set_ylabel('Y = Dealer')
        axes.set_zlabel('Z = Value Function')
        plt.show()


if __name__ == '__main__':
    agent = MonteCarloControl()
    agent.train()
    agent.plot_optimal_value_function()
