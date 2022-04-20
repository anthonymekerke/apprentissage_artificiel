import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten

def load_real_samples(digit):
    (trainX, trainY), (_, _) = mnist.load_data()
    filter = np.where(trainY == digit)
    trainX = trainX[filter]
    X = trainX.astype('float32')
    X = X/255.0
    return X

def create_generator(latent_space_dim=100, n_outputs = 28*28):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=latent_space_dim))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    return model

def generate_real_samples(data, n_samples):
    indices = np.random.randint(0, data.shape[0], n_samples)
    X = data[indices]
    X = X.reshape(n_samples, 784)
    y = np.ones((n_samples, 1))
    return X, y

def generate_latent_points(latent_dim=100, n=100):
    x_input = np.random.normal(0, 1, latent_dim*n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

def generate_fake_samples(generator, latent_dim=100, n=100):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    y = np.zeros((n,1))
    return X, y

def create_discriminator(input_size=28*28):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=input_size))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def show_images(examples, n=10):
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        tmp = examples[i].reshape(28,28)
        plt.imshow(tmp, cmap='gray_r')
    plt.show()

def summarize_performances(epoch, generator, discriminator, dataset, latent_dim=100, n_points=1000):
    X_real, y_real = generate_real_samples(dataset, n_points)
    X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_points)
    _, acc_real = discriminator.evaluate(X_real, y_real)
    _, acc_fake = discriminator.evaluate(X_fake, y_fake)
    print('Epoch {}, fake acc {}, real acc {}'.format(epoch, acc_fake, acc_real))
    show_images(X_fake)

def train_gan(generator, discriminator, gan, dataset, latent_dim=100, n_epochs=150, n_batch=512, n_eval=100):
    for i in range(n_epochs):
        X_real, y_real = generate_real_samples(dataset, n_batch//2)
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_batch//2)
        discriminator.train_on_batch(X_real, y_real)
        discriminator.train_on_batch(X_fake, y_fake)
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        gan.train_on_batch(x_gan, y_gan)
        if (i % n_eval == 0):
            summarize_performances(i, generator, discriminator, dataset, latent_dim)

def main():
    discriminator = create_discriminator()
    generator = create_generator()

    gan = create_gan(generator, discriminator)
    train_gan(generator, discriminator, gan)

if __name__=="__main__":
    main()