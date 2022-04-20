import numpy as np
import random as rd

from Treasure_Hunt import TreasureHunt
from StateActionMatrix import StateActionMatrix


class Agent:
    def __init__(self, gamma, nb_state, state_action_matrix):
        self.gamma = gamma
        self.state_action_matrix = state_action_matrix
        # self.q_learning = np.zeros(nb_state, dtype={'names': ['state', 'actions'], 'formats': ['2int8', '4float32']})
        self.q_learning = StateActionMatrix(nb_state)

    def transition(self, state, action, state_action_matrix):
        next_state = self.treasure_hunt.get_next_state(state, action, state_action_matrix)

        max_score = self.treasure_hunt.get_max_score_state(next_state, state_action_matrix)

        self.q_learning[state][1][action] = state_action_matrix[state][1][action] + (self.gamma * max_score)

        print(self.q_learning)

        return next_state

    def episode(self):
        state_action_matrix = self.treasure_hunt.get_matrix_state_action()

        current_state = rd.randrange(self.nb_state)

        possible_action = self.treasure_hunt.get_possible_actions(current_state, state_action_matrix)
        current_action = rd.choice(possible_action)

        index = 0
        while index < 5:
            print('transition ' + str(index))
            index += 1
            current_state = self.transition(current_state, current_action, state_action_matrix)

            possible_action = self.treasure_hunt.get_possible_actions(current_state, state_action_matrix)
            current_action = rd.choice(possible_action)
