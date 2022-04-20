#!usr/bin/python3

import sys
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

nomFichier = sys.argv[1]
fichier = open(nomFichier, "r")

lesLignes = fichier.readlines()
fichier.close()

xclasse1 = []
yclasse1 = []
xclasse2 = []
yclasse2 = []

for uneLigne in lesLignes:
    split = uneLigne.split()

    if int(split[2]) > 0:
        xclasse1.append(float(split[0]))
        yclasse1.append(float(split[1]))
    else:
        xclasse2.append(float(split[0]))
        yclasse2.append(float(split[1]))


fig = plt.figure()
plt.plot(xclasse1, yclasse1, 'bo', xclasse2, yclasse2, 'rx')

plt.show()
