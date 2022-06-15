import numpy as np
from rl_base import Agent, ActionControl
import os


class QAgent(Agent):

    def __init__(self, n_states, n_actions, action_control: ActionControl, name='QAgent', lr=0.1, gamma=0.99,
                 initial_q_value=0.0, q_table=None):
        super().__init__(name)
        self.gamma = gamma
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)
        self.action_control = action_control

    def init_q_table(self, initial_q_value=0.):
        q_table = initial_q_value * np.ones((self.n_states, len(self.action_space)))
        return q_table

    def choose_action(self, observation):
        assert 0 <= observation < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        return self.action_control.get_action(self, observation)

    # TODO q_table powinno zostać zaktualizowane tutaj
    def learn(self, observation, action, reward, new_observation, done):
        self.q_table[observation, action] = (1-self.lr) * self.q_table[observation, action] + \
                                            self.lr*(reward+self.gamma*max(self.q_table[new_observation, :]))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def from_state_to_idx(self, state):
        return state

    def observe(self, state):
        return self.from_state_to_idx(state)

    def get_instruction_string(self):
        return self.action_control.get_instruction_string()
