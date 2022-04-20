import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPooling2D

def load_dataset(path, img_size=150):
    data = []
    labels = []

    for subpath, _, files in os.walk(path):
        tmp = subpath.split('/')
        cl = tmp[-1] # on recupère le dernier élément de la liste
        if len(files):
            for f in files:
                temp = cv2.imread(os.path.join(subpath, f))
                temp = cv2.resize(temp, (img_size, img_size)).astype('float32') / 255.
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                data.append(temp)
                labels.append(cl)
    data = np.array(data)
    labels = np.array(labels)
    print(labels)
    label_encoder = LabelEncoder()
    vect_labels = label_encoder.fit_transform(labels)
    print(vect_labels)
    labels = to_categorical(vect_labels)
    num_classes = labels.shape[1]

    return (data, labels, num_classes, label_encoder)

def split_data(data, labels, test_ratio):
    return train_test_split(data, labels, test_ratio)

def create_model(input_shape, num_classes):
    model = Sequential()
    
    #Partie convolutionnelle
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
    #model.add(Conv2D(32, (3,3), activation='relu'))
    #model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Partie fully connected
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'loss'])

    return model

def show_results(model, X_test, Y_test, label_encoder):
    print(model.evaluate(X_test, Y_test))

    y_pred = model.predict(X_test)

    for i in range(X_test.shape[0]):
        plt.imshow(X_test[i])
        predicted_label = np.argmax(y_pred[i])
        predicted_value = np.max(y_pred[i])
        true_label = np.argmax(y_test[i])

        color = 'red'
        if predicted_label == true_label:
            color = 'blue'

        plt.xlabel("{}: {:2.0f}%, ({})".format(label_encoder.inverse_transform([predicted_label]),
        100 * predicted_value, label_encoder.inverse_transform([true_label]), color=color))

        plt.show()

def main():
    test_ratio = 0.3
    training = True
    epoch = 10
    batch_size = 16
    img_size = 64

    path = 'simpsons_dataset/'
    data, labels, num_classes, label_encoder = load_dataset(path)

    x_train, x_test, y_train, y_test = split_data(data, labels, test_ratio)

    model = create_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)

    if training:
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size)
        model.save('NN_simpsons')
    else:
        model = load_model('NN_simpsons')

    show_results()

if __name__ == '__main__':
    main()