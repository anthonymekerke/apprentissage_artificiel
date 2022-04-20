import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten

def generate_real_points(digit=2):
    mnist = keras.datasets.mnist
    (img, label), _ = mnist.load_data()
    filter_array = []

    for i in range(label.shape[0]):
        if(label[i] == digit):
            filter_array.append(True)
        else:
            filter_array.append(False)

    # possibilité de remplacer la boucle par:
    # filter_array = np.where(label == digit)

    x_real = img[filter_array]
    y_real = label[filter_array]

    return x_real, y_real

def generate_fake_points(n=100):
    x_fake = np.random.randint(0, 255, size=(n, 28, 28))
    y_fake = np.zeros((n, 1))

    return x_fake, y_fake

def create_discriminator(input_size=784):
    model = Sequential()
    model.add(Dense(128, activation='relu'), input_dim=input_size)
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_generator(latent_space_dim=5, n_outputs=784):
    model = Sequential()
    model.add(Dense(15, activation='relu', input_dim=latent_space_dim))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def generate_latent_points(latent_dim=5, n=100):
    x_input = np.random.normal(0, 1, latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)

    return x_input

def generate_fake_samples(generator, latent_dim=5, n=100):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    y = np.zeros((n,1))

    return X, y

def generate_real_samples(data, n):
    indices = np.random.randint(0, data.shape[0], n) # les n indices aléatoires
    X = data[indices] # on prend les images correspondantes
    X = X.reshape(n, 784) # transforme (n, 28,28) en (n, 784)
    y = np.ones((n, 1)) # n Vraies images, on leur donne le label 1 
    
    return X, y

def create_gan(generator, discriminator):
    discriminator.trainable = False #discriminateur seulement utilisé pour distingué vrai du faux
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def train_gan(generator, discriminator, gan, latent_dim=5, n_epochs=3000, n_batch=256, n_eval=1000):
    img, _ = generate_real_points(2)
    for i in range(n_epochs):
        X_real, y_real = generate_real_samples(img, n_batch//2)
        X_fake, y_fake = generate_fake_samples(generator,latent_dim, n_batch//2)

        discriminator.train_on_batch(X_real, y_real) # similaire si train un seul vector
        discriminator.train_on_batch(X_fake, y_fake) # (ne modifie pas la struct du reseaux neurone)

        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        gan.train_on_batch(x_gan, y_gan)
        
        if (i % n_eval == 0):
            summarize_performances(i, generator, discriminator, latent_dim)

def summarize_performances(epoch, generator, discriminator, latent_dim=5, n_points=100):
    X_real, y_real = generate_real_points(n_points)
    X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_points)

    _, acc_real = discriminator.evaluate(X_real, y_real)
    _, acc_fake = discriminator.evaluate(X_fake, y_fake)

    print('Epoch {}, fakeAcc {}, realAcc {}'.format(epoch, acc_fake, acc_real))

    plt.scatter(X_real[:,0], X_real[:,1], color='blue')
    plt.scatter(X_fake[:,0], X_fake[:,1], color='red')
    plt.legend(['Real points', 'Fake points'])
    #filename = 'images/generated/cos{}_'.format(epoch)
    #plt.savefig(filename)
    plt.show()

def main():
    #x_real, y_real = generate_real_points()
    #x_fake, y_fake = generate_fake_points(x_real.shape[1])

    '''
    discriminator = create_discriminator()
    generator = create_generator()

    gan = create_gan(generator, discriminator)
    train_gan(generator, discriminator, gan)
    '''

if __name__=='__main__':
    main()