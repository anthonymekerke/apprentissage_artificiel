import random as rd
from Treasure_Hunt import TreasureHunt


def e_greedy(th):
    init_pos = [rd.randrange(th.lines), rd.randrange(th.columns)]
    result = th.init_pirate(init_pos)

    steps = 0

    if result == 10:
        # game won at initialization
        return steps

    while result != th.treasure_value:
        steps += 1

        possible_moves = th.get_possible_moves(th.pirate)

        random = rd.random()

        if random > 0.1:
            best_move = th.best_possible_move(possible_moves)
            result = th.move_pirate(best_move)
        else:
            index = rd.randrange(len(possible_moves))
            random_move = possible_moves[index]
            result = th.move_pirate(random_move)

        if result == 10:
            return steps


def n_episode(th, n):
    nb_episode = 0
    average_step = 0

    while nb_episode < n:
        nb_episode += 1
        average_step += e_greedy(th)

    return float(average_step / n)
