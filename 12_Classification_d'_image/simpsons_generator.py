
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Question 1: Il y a 8 classes

# Question 2: Il y a 10 061 exemples

# Question 3: 

# Question 4:

def analyze_data(path):
    total = 0
    for subpath, directories, files in os.walk(path):
        if len(files):
            print('classe {} nb exemples: {}'.format(subpath, len(files)))
            total += len(files)
        else:
            print('nb classes: {}'.format(len(directories)))
    print('nb total exemples: {}'.format(total))

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def load_generators(path, batch_size, ratio):
    data_generator = ImageDataGenerator(rescale=1./255., validation_split=ratio)
    train_generator = data_generator.flow_from_directory(
        directory=path,
        target_size=(150,150), shuffle=False, class_mode='categorical',
        batch_size=batch_size, subset='training'
    )
    validation_generator = data_generator.flow_from_directory(
        directory=path,
        target_size=(150,150), shuffle=False, class_mode='categorical',
        batch_size=batch_size, subset='validation'
    )

    return (train_generator, validation_generator)

def main():
    path = 'simpsons_dataset/'
    analyze_data(path)
    input_shape=(150,150, 3)
    model = create_model(input_shape)
    validation_ratio = 0.25
    samples = 10061
    nb_validation_samples = samples * validation_ratio
    nb_train_samples = samples - nb_validation_samples
    batch_size = 32
    train_generator, validation_generator = load_generators(path, batch_size, validation_ratio)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size, #7550 images de train, batch de 32 --> 7550 //32 = 235
        epochs=4,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
    )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])
    plt.show()

if __name__ == '__main__':
    main()