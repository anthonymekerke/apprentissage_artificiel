import numpy as np
import random as rd

from Treasure_Hunt import TreasureHunt
from State_Action_Matrix import StateActionMatrix


class Agent:
    def __init__(self, gamma, nb_state, state_action_matrix):
        self.gamma = gamma
        self.state_action_matrix = state_action_matrix
        self.q_learning = StateActionMatrix(nb_state, '4float32')

        self.q_learning.matrix['state'] = self.state_action_matrix.matrix['state'].copy()

    def transition(self, state, action):
        next_state = self.state_action_matrix.get_next_state(state, action)
        max_score = self.state_action_matrix.get_max_score_state(next_state)

        self.q_learning.matrix[state][1][action] = self.state_action_matrix.matrix[state][1][action] + (self.gamma * max_score)

        return next_state

    def episode(self):
        # select an initial random state
        nb_state = self.state_action_matrix.nb_state
        current_state = rd.randrange(nb_state)

        # select an initial random action
        possible_actions = self.state_action_matrix.get_possible_actions(current_state)
        current_action = rd.choice(possible_actions)

        finished = False
        steps = 0
        while not finished:
            steps += 1
            previous_state = current_state
            current_state = self.transition(previous_state, current_action)

            if self.q_learning.get_max_score_state(previous_state) >= 10:
                finished = True

            possible_actions = self.state_action_matrix.get_possible_actions(current_state)
            current_action = rd.choice(possible_actions)

        return steps

    def n_episode(self, n):
        nb_episode = 0
        average_step = 0

        while nb_episode < n:
            nb_episode += 1
            average_step += self.episode()

        return float(average_step/n)
