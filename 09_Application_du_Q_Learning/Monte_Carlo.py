import random as rd
from Treasure_Hunt import TreasureHunt


class MonteCarlo:

    def __init__(self, treasure_hunt):
        self.environment = treasure_hunt

    @staticmethod
    def random_possible_move(list_moves):
        index = rd.randrange(len(list_moves))

        return list_moves[index]

    def random_step(self, position, nb_step_max):
        nb_step = 0
        result = self.environment.init_pirate(position)

        while nb_step < nb_step_max and result != self.environment.treasure_value:
            nb_step += 1

            possible_moves = self.environment.get_possible_moves(self.environment.pirate)
            next_move = self.random_possible_move(possible_moves)
            result = self.environment.move_pirate(next_move)

        return nb_step_max - nb_step

    def monte_carlo(self, nb_step_max):
        best_score = -1
        steps = 0
        position = self.environment.random_position()

        while best_score != nb_step_max:
            steps += 1

            possible_moves = self.environment.get_possible_moves(position)

            for i in range(len(possible_moves)):
                score = self.random_step(possible_moves[i], nb_step_max)

                if score > best_score:
                    best_score = score
                    position = possible_moves[i]

        return steps

    def n_episode(self, nb_step_max, n):
        nb_episode = 0
        average_steps = 0

        while nb_episode < n:
            nb_episode += 1
            average_steps += self.monte_carlo(nb_step_max)

        return float(average_steps/n)
