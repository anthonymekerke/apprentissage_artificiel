import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

num_train, img_rows, img_cols = train_images.shape
depth = 1
nb_classes = 10
nb_epoch = 5
batch_size = 32

train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, depth)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, depth)

print(train_images.shape)


input_shape = (img_rows, img_cols, depth)
nb_filters = 32
pool_size = (2,2)
kernel_size = (3,3)

model = keras.Sequential()
model.add(keras.layers.Conv2D(nb_filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(nb_classes, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.summary()

model.fit(train_images, train_labels, 
    batch_size=batch_size, epochs=nb_epoch, verbose=1,
    validation_data=(test_images, test_labels)
)

score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
