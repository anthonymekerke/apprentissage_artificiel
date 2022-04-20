#!usr/bin/python3

import numpy as np
import constant as const

x1 = np.random.uniform(0, 1, const.nb_example)
x2 = np.random.uniform(0, 1, const.nb_example)

"""
#question 1
tag = x1 + x2 - 1
tag = [1 if (x > 0) else -1 for x in tag]
"""

#question 3.4
tag = []
for i in range(len(x1)):
	compute = x1[i] + x2[i] - 1
	
	if(compute > 0 and x2[i] > 0.5):
		tag.append(1)
	else:
		tag.append(-1)


line = [x1, x2, tag]

nb_example = const.nb_example
share_test = const.share_test

train_set = open('train.csv', 'w')
test_set = open('test.csv', 'w')

for i in range(int(nb_example * (1 - share_test))):
    train_set.write(str(x1[i]) + ' ' + str(x2[i]) + ' ' + str(tag[i]) + '\n')

for i in range(int(nb_example * share_test)):
    test_set.write(str(x1[i]) + ' ' + str(x2[i]) + ' ' + str(tag[i]) + '\n')

train_set.close()
test_set.close()
