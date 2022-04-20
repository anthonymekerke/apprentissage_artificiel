#!usr/bin/python3

import math
import random as rd
from Manchot import Manchot

def strategie_aleatoire(nbIteration, manchots):
    gain = 0.0

    for i in range(nbIteration):
        index = rd.randint(0, len(manchots) - 1)
        gain += manchots[index].tirerBras()

    return int(gain)

def strategie_gloutonne(nbIteration, manchots):
    prev_gain = 0.0
    gain = 0.0
    manchot_best = 0

    for i in range(len(manchots)):
        current_gain = manchots[i].tirerBras()

        if prev_gain < current_gain:
            prev_gain = current_gain
            manchot_best = i

        gain += current_gain

    for i in range(nbIteration - len(manchots)):
        gain += manchots[manchot_best].tirerBras()

    return int(gain)

def rechercheUCB(nbIteration, manchots, K):
    nb_pull_total = 0
    nb_pull = [0] * len(manchots)
    sum_gain = [0] * len(manchots)

    for i in range(len(manchots)):
        sum_gain[i] += manchots[i].tirerBras()
        nb_pull[i] += 1
        nb_pull_total += 1

    while nb_pull_total < nbIteration:
        nb_pull_total += 1
        manchot_best = 0
        UCB_best = 0

        for i in range(len(manchots)):
            UCB_current = (sum_gain[i] / nb_pull[i]) + K * math.sqrt(math.log2(nb_pull[i]) / nb_pull_total)

            if UCB_current > UCB_best:
                manchot_best = i
                UCB_best = UCB_current

        sum_gain[manchot_best] += manchots[manchot_best].tirerBras()
        nb_pull[manchot_best] += 1

    print("Repartition essais UCB:" + str(nb_pull))

    gain = 0
    for i in range(len(manchots)):
        gain += sum_gain[i]

    return int(gain)

def creerManchots(nb):

    manchots = []

    for i in range(nb):
        esperance = rd.uniform(-10, 10)
        variance = rd.uniform(0, 10)

        m = Manchot(esperance, variance)
        manchots.append(m)

    return manchots

def main():
    nb_iter = 15000
    K = math.sqrt(2)
    manchots = creerManchots(30)

    print("strategie random: " + str(strategie_aleatoire(nb_iter, manchots)))
    print("strategie gloutonne: " + str(strategie_gloutonne(nb_iter, manchots)))
    print("score UCB: " + str(rechercheUCB(nb_iter, manchots, K)))

# ctrl + alt + S -> save as image

if __name__ == '__main__':
    main()
