import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten

#fonction retournant les ensembles nécessaire à l'apprentissage.
#param:
    #noise=0.5 -> Intensité du bruit appliqué aux images
#returns:
    #X_train_noised -> ensemble d'apprentissage X: image bruité
    #X_train -> ensemble d'apprentissage Y: image en clair
    #X_test_noised -> ensemble de test X: image bruité
    #X_test-> ensemble de test Y: image en clair
def prepare_data(noise=0.5):
    mnist = keras.datasets.mnist
    (X_train, _), (X_test, _) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train_noised = X_train + noise * np.random.normal(0., 1., size=X_train.shape)
    X_test_noised = X_test + noise * np.random.normal(0., 1., size=X_test.shape)

    return (X_train_noised, X_train, X_test_noised, X_test)

def show_images(examples, n=4):
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        tmp = examples[i].reshape(28,28)
        plt.imshow(tmp, cmap='gray_r')
    plt.show()

#Fonction créant l'encoder
#param:
    #input_shape=(28*28): dimension des images
#return:
    #model -> encoder: a pour role de "compresser" les images fournits en entrée
def create_encoder(input_shape=(28*28)):
    model = Sequential()

    #Ajout d'une couche 'Dense'
        #activation='relu': efficace pour distinguer les valeurs
    model.add(Dense(128, input_dim=input_shape, activation='relu'))

    #Ajout d'une couche 'Dense'
        #activation='relu': efficace pour distinguer les valeurs
    model.add(Dense(64, activation='relu'))

    #Ajout d'une couche 'Dense' de sortie
        #ouput_shape=32: valeurs faibles pour 'compresser' l'image
    model.add(Dense(32, activation='relu'))

    return model
#Fonction créant le decoder
#param:
    #input_shape=32: sortie du réseaux 'encoder'
#return:
    #model -> decoder: il est construit avec les mêmes couches que 'encoder' mais "monté à l'envers"
def create_decoder(input_shape=(32)):
    model = Sequential()

    #Ajout d'une couche 'Dense' "symetrique" par rapport au model 'encoder'
    model.add(Dense(64, input_dim=input_shape, activation='relu'))

    #Ajout d'une couche 'Dense' "symetrique" par rapport au model 'encoder'
    model.add(Dense(128, activation='relu'))

    #Ajout d'une couche 'Dense' "symetrique" par rapport au model 'encoder'
        #activation='sigmoid': utilisation d'une sigmoid pour 'lisser' les valeurs
    model.add(Dense(28*28, activation='sigmoid'))

    return model

def train_autoencoder(encoder, decoder, nb_epochs, X_train, y_train, batch_size):
    autoencoder = Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

    autoencoder.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size)

    return autoencoder


def main():
    X_train, y_train, X_test, y_test = prepare_data()

    encoder = create_encoder()
    decoder = create_decoder()

    autoencoder = train_autoencoder(encoder, decoder, 10, X_train, y_train, 256)
    encoded_img = encoder.predict(X_test)
    decoded_img = decoder.predict(encoded_img)

    show_images(X_test)
    show_images(decoded_img)

if __name__=="__main__":
    main()