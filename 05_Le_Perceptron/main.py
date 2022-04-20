#!/usr/bin/Python3

import sys
import constant as const
import matplotlib.pyplot as plt
from Neurone import Neurone

filename_train = sys.argv[1]
filename_test = sys.argv[2]

file_train = open(filename_train, "r")
lines_train = file_train.readlines()
file_train.close()

file_test = open(filename_test, 'r')
lines_test = file_test.readlines()
file_test.close()

x1_train = []
x2_train = []
etiquette_train = []

x1_test = []
x2_test = []
etiquette_test = []

for line in lines_train:
    split = line.split()

    x1_train.append(float(split[0]))
    x2_train.append(float(split[1]))
    etiquette_train.append(float(split[2]))
    
for line in lines_test:
    split = line.split()

    x1_test.append(float(split[0]))
    x2_test.append(float(split[1]))
    etiquette_test.append(float(split[2]))

train_size = len(etiquette_train)
test_size = len(etiquette_test)
model = Neurone(const.learning_step)
list_nb_errors = []

for i in range(100):
    nb_errors = 0
    for j in range(train_size):
        example = [x1_train[j], x2_train[j]]
        model.compute_output(example)

        if model.output != etiquette_train[j]:
            nb_errors += 1
            model.update(example, etiquette_train[j])

    list_nb_errors.append(nb_errors)
    
nb_errors_test = 0
for i in range(test_size):
        example = [x1_test[i], x2_test[i]]
        model.compute_output(example)

        if model.output != etiquette_test[i]:
            nb_errors_test += 1

print("errors with validation: " + str(nb_errors_test) + "/" + str(test_size))

fig = plt.figure()
plt.title("errors train set: nb example= " + str(train_size) +  ", learning step= " + str(const.learning_step))
plt.plot(list_nb_errors, 'b')
plt.show()
