import numpy as np


class StateActionMatrix:

    def __init__(self, nb_state):
        self.matrix = np.zeros(nb_state, dtype={'names': ['state', 'actions'], 'formats': ['2int8', '4int8']})

    def get_next_state(self, state, action):
        if self.matrix[state][1][action] == -1:
            return -1

        position = self.matrix[state][0].copy()

        if action == 0:
            position[0] -= 1

        if action == 1:
            position[0] += 1

        if action == 2:
            position[1] += 1

        if action == 3:
            position[1] -= 1

        for i in range(len(self.matrix['state'])):
            if (position == self.matrix[i][0]).all():
                return i

        return -1

    def get_possible_actions(self, state):
        possible_actions = []

        for i in range(4):
            if self.matrix[state][1][i] != -1:
                possible_actions.append(i)

        return possible_actions

    def get_max_score_state(self, state):
        max = 0

        for i in range(4):
            if max < self.matrix[state][1][i]:
                max = self.matrix[state][1][i]

        return max
