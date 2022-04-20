import random as rd
import numpy as np

from State_Action_Matrix import StateActionMatrix


class TreasureHunt:
    def __init__(self, li, co, b):
        self.columns = co
        self.lines = li
        self.bottles = b
        self.pirate = []
        self.treasure_value = 10
        self.island = np.zeros((li, co))

        i = 0
        while i < self.bottles + 1:
            l = rd.randrange(self.lines)
            c = rd.randrange(self.columns)

            if self.island[l][c] == 0:
                self.island[l][c] = 2
                i = i + 1

            if i > self.bottles:
                self.island[l][c] = 10

    def set_case(self, pos, case):
        self.island[pos[0]][pos[1]] = case

    def get_case(self, pos):
        if (pos[0] < 0) or (pos[0] >= self.lines) or (pos[1] < 0) or (pos[1] >= self.columns):
            return -1

        return self.island[pos[0]][pos[1]]

    def random_position(self):

        return [rd.randrange(self.lines), rd.randrange(self.columns)]

    def init_pirate(self, pos):
        result = self.get_case(pos)
        self.pirate = pos

        return result

    def move_pirate(self, pos):
        result = self.get_case(pos)
        self.pirate = pos

        return result

    def get_possible_moves(self, position):
        list_moves = []

        # right position
        if position[1] < self.columns - 1:
            list_moves.append([position[0], position[1]+1])

        # left position
        if position[1] > 0:
            list_moves.append([position[0], position[1]-1])

        # up position
        if position[0] > 0:
            list_moves.append([position[0] -1, position[1]])

        # down position
        if position[0] < self.lines -1:
            list_moves.append([position[0]+1, position[1]])

        return list_moves

    def best_possible_move(self, list_moves):
        best = []
        score_best = 0
        for i in range(len(list_moves)):
            if self.get_case(list_moves[i]) > score_best:
                score_best = self.get_case(list_moves[i])
                best = list_moves[i]

        return best

    def get_matrix_state_action(self):
        nb_state = self.lines * self.columns
        state_action = StateActionMatrix(nb_state, '4int8')

        i = 0
        for li in range(self.lines):
            for co in range(self.columns):
                position = [li, co]
                state_action.matrix[i][0] = position

                north = [position[0]-1, position[1]]
                state_action.matrix[i][1][0] = self.get_case(north)

                south = [position[0]+1, position[1]]
                state_action.matrix[i][1][1] = self.get_case(south)

                east = [position[0], position[1] + 1]
                state_action.matrix[i][1][2] = self.get_case(east)

                west = [position[0], position[1]-1]
                state_action.matrix[i][1][3] = self.get_case(west)

                i = i + 1

        return state_action
