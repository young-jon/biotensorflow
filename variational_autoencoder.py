from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.objectives import binary_crossentropy

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

def main():
    (x_train, _), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    hidden_layer_size = 256
    latent_dim = 2
    batch_size = 100
    epochs = 50

    # Build encoder
    x = Input(batch_shape=(batch_size, 784))
    h_q = Dense(hidden_layer_size, activation="relu")(x)
    z_mean = Dense(latent_dim)(h_q)
    z_log_var = Dense(latent_dim)(h_q)

    # Reparameterization (sampling from Guassian dist defined by encoder)
    def sample(args):
        mean, log_var = args
        epsilon = K.random_normal(
            shape=(batch_size, latent_dim), mean=0., stddev=1)
        return mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sample)([z_mean, z_log_var])

    # Build decoder
    decoder_hidden_layer = Dense(hidden_layer_size, activation="relu")
    decoder_output_layer = Dense(784, activation="sigmoid")

    h_p = decoder_hidden_layer(z)
    decoder_output = decoder_output_layer(h_p)

    # VAE
    vae = Model(x, decoder_output)

    # Encoder model
    encoder = Model(x, z_mean)

    # Generator model
    generator_x = Input(shape=(latent_dim,))
    generator_h = decoder_hidden_layer(generator_x)
    generator_output = decoder_output_layer(generator_h)
    generator = Model(generator_x, generator_output)

    def vae_loss(y_true, y_pred):
        reconstruction_error = binary_crossentropy(y_pred, y_true)
        kl_divergence = - 0.5 * K.mean(
            1 + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=-1)
        return reconstruction_error + kl_divergence

    vae.compile(optimizer="adam", loss=vae_loss)
    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))

    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

    decoded_imgs = generator.predict(x_test_encoded)
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()