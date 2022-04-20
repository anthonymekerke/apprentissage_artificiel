#!/usr/bin/python3

import random as rd
from Agent import Agent
from Treasure_Hunt import TreasureHunt

def e_greedy(th):
    init_pos = [rd.randrange(th.lines), rd.randrange(th.columns)]
    result = th.init_pirate(init_pos)

    if result == 10:
        return 'game won at initialization'

    nb_coup = 20

    while nb_coup >= 0:
        nb_coup = nb_coup - 1

        possible_moves = th.get_possible_moves(th.pirate)

        random = rd.random()

        if random > 0.1:
            best_move = th.best_possible_move(possible_moves)
            result = th.move_pirate(best_move)
        else:
            random_move = th.random_possible_move(possible_moves)
            result = th.move_pirate(random_move)

        if result == 10:
            return 'game won'

    return 'game lost'


def main():
    treasure_hunt = TreasureHunt(4, 2, 1)
    treasure_hunt.init_pirate([0, 0])
    print('Island:')
    print(treasure_hunt.island)

    matrix = treasure_hunt.get_matrix_state_action()
    print('state<->action matrix:')
    print(matrix)

    agent = Agent(0.9, treasure_hunt)
    agent.episode()


if __name__ == '__main__':
    main()
