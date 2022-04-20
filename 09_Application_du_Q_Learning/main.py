#!/usr/bin/python3

import E_greedy
from Q_Learning import Agent
from State_Action_Matrix import StateActionMatrix
from Monte_Carlo import MonteCarlo
from Treasure_Hunt import TreasureHunt


def test(nb_run, size, nb_bottles):
    treasure_hunt = TreasureHunt(size[0], size[1], nb_bottles)

    monte_carlo = MonteCarlo(treasure_hunt)

    matrix = treasure_hunt.get_matrix_state_action()
    q_learning = Agent(0.9, matrix.nb_state, matrix)

    print(str(nb_run) + " xps, environment size " + str(size[0]) + "x" + str(size[1]) + ", " + str(
        nb_bottles) + " local optima")

    steps_q_learning = q_learning.n_episode(nb_run)
    steps_monte_carlo = monte_carlo.n_episode(nb_step_max=10, n=nb_run)
    steps_e_greedy = E_greedy.n_episode(treasure_hunt, nb_run)

    print("Q learning: " + str(steps_q_learning) + " steps (" + " s per run)")
    print("monte carlo: " + str(steps_monte_carlo) + " steps (" + " s per run)")
    print("e greedy: " + str(steps_e_greedy) + " steps (" + " s per run)")


def main():

    nb_run = 10
    size = [3, 5]
    local_optimum = 10
    test(nb_run, size, local_optimum)

    size = [5, 7]
    test(nb_run, size, local_optimum)


if __name__ == '__main__':
    main()
