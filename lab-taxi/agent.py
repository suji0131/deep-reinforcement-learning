import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, no_of_actions=6):
        """ Initialize agent.

        Params
        ======
        - no_of_actions: number of actions available to the agent
        """
        self.no_of_actions = no_of_actions
        self.Q = defaultdict(lambda: np.zeros(self.no_of_actions))

    def select_action(self, state, epsilon=0.05) -> int:
        """ Given the state, select an action. Using e-greedy policy.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # generate which policy to
        # choose "greedy"(exploitation) or "non-greedy"(exploration)
        policy_probs = [1 - epsilon, epsilon]  # ["greedy"(exploitation), "non-greedy"(exploration)]
        policy_method = np.random.choice(np.arange(2), p=policy_probs)

        if policy_method == 0:  # "greedy"(exploitation)
            action = np.argmax(self.Q[state])
        elif policy_method == 1:  # "non-greedy"(exploration)
            probs = [1 / self.no_of_actions for _ in range(self.no_of_actions)]
            action = np.random.choice(np.arange(self.no_of_actions), p=probs)
        else:
            raise ValueError("Policy generated is out-of-bounds.")

        return action

    def step(self, state, action, reward, next_state, done, gamma=1.0, alpha=.01):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            expected_return_next_step = 0
        else:
            # SarsaMax or Q-learning.
            expected_return_next_step = np.max(self.Q[next_state])

        self.Q[state][action] += alpha * (
            reward
            + (gamma * expected_return_next_step)
            - self.Q[state][action]
        )
        return 0
