import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# load data from mnist library
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print shape of datas
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# le modèle 'sequential' est un ensemble linéaire de couches
model = keras.Sequential([
    # transforme une matrice 28x28 en un tableau de 784
    keras.layers.Flatten(input_shape=(28,28)),
    # couche entièrement connectée de 128 neurones
    keras.layers.Dense(128, activation=tf.nn.relu),
    # couche entièrement connectée de 10 neurones
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    # optimisation = descente de gradient stochastique
    optimizer='sgd',
    # perte = entropie croisée
    loss='sparse_categorical_crossentropy',
    # mesure performance = accuracy
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("perte: {}, accuracy: {}".format(test_loss, test_acc))
