import math
import random as rd

class Manchot:
    def __init__(self, mu, sigma):
        self.esperance = mu
        self.variance = sigma

    def tirerBras(self):
        rand1 = rd.random()
        rand2 = rd.random()

        return self.variance * math.sqrt(-2.0 * math.log2(rand1)) + math.cos(2.0 * math.pi * rand2) + self.esperance
