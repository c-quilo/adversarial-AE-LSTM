from __future__ import print_function, division

import tensorflow.keras as tf
import tensorflow
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as backend
import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
from keras.constraints import Constraint
from keras.initializers import RandomNormal

from keras import optimizers
from keras.utils import np_utils
import tensorflow.keras as tf

class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# clip model weights to a given hypercube
class AAE():

    def __init__(self, directory_data, field_name, npcs, initNNodes, latent_dim, GANorWGAN):

        # Wasserstein loss
        def wasserstein_loss(y_true, y_pred):
            return backend.mean(y_true * y_pred)

        self.field_name = field_name
        self.directory_data = directory_data
        self.latent_dim = latent_dim
        self.npcs = npcs
        self.constraint = 0.01
        self.dropoutNumber = 0.5
        self.alpha = 0.3
        self.initNNodes = initNNodes
        self.GANorWGAN = GANorWGAN

        self.c1_hist = []
        self.c2_hist = []
        self.g_hist = []

        self.optimizer = tf.optimizers.Nadam()

        if self.GANorWGAN == 'WGAN':
            self.loss = wasserstein_loss
        elif self.GANorWGAN == 'GAN'
            self.loss = 'binary_crossentropy'

        self.loss_gen = 'mse'

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the encoder and decoder
        self.generator_encoder = self.build_generator_encoder()
        self.generator_decoder = self.build_generator_decoder()

        # Only the generator is trained through the combined model, thus:
        self.discriminator.trainable = False

        # Connecting models
        real_input = tf.Input(shape=self.npcs)
        encoder_output = self.generator_encoder(real_input)
        decoder_output = self.generator_decoder(encoder_output)
        discriminator_output = self.discriminator(encoder_output)

        # The combined model stacks the autoencoder and discriminator
        # The stacked model has one input and two outputs: the decoded input and the discriminator output
        self.combined = tf.Model(real_input, [decoder_output, discriminator_output], name = 'AAE')
        self.combined.compile(loss=[self.loss_gen, self.loss], loss_weights=[0.999, 0.001], optimizer=self.optimizer)

    def build_discriminator(self):
        init = RandomNormal(stddev=0.02)
        const = ClipConstraint(0.01)

        in_disc = tf.Input(shape=(self.latent_dim))
        disc = tf.layers.LeakyReLU(self.alpha)(in_disc)
        disc = tf.layers.BatchNormalization()(disc)
        disc_output = tf.layers.Dense(1, activation='sigmoid')(disc)
        discriminator = tf.Model(in_disc, disc_output, name='Discriminator')
        discriminator.compile(loss=self.loss, optimizer=self.optimizer)

        return discriminator

    def build_generator_encoder(self):
        init = RandomNormal(stddev=0.02)
        init = tf.initializers.RandomNormal(stddev=0.02)

        input_enc = tf.Input(shape=self.npcs)
        nNodes = self.initNNodes
        flag = 0
        while nNodes > latent_dim:
            if flag == 0:
                enc = tf.layers.Dense(nNodes)(input_enc)
                flag = 1
            else:
                enc = tf.layers.Dense(nNodes)(enc)
            enc = tf.layers.LeakyReLU(self.alpha)(enc)
            enc = tf.layers.BatchNormalization()(enc)
            nNodes = nNodes / 2
        mu = tf.layers.Dense(latent_dim)(enc)
        sigma = tf.layers.Dense(latent_dim)(enc)

        # The latent representation ("fake") in a Gaussian distribution is then compared to the "real" arbitrary Gaussian
        # distribution fed in the Discriminator
        latent_repr = tf.layers.Lambda(
            lambda p: p[0] + backend.random_normal(backend.shape(p[0])) * backend.exp(p[1] / 2))(
            [mu, sigma])
        generator_encoder = tf.Model(input_enc, latent_repr, name='Encoder')
        generator_encoder.summary()
        return generator_encoder

    def build_generator_decoder(self):
        init = RandomNormal(stddev=0.02)
        init = tf.initializers.RandomNormal(stddev=0.02)

        # Input to the decoder is the latent space from the encoder
        input_dec = tf.Input(shape=self.latent_dim)
        n = 2 * latent_dim
        flag = 0
        while n <= self.initNNodes:
            if flag == 0:
                dec = tf.layers.Dense(n)(input_dec)
                flag = 1
            else:
                dec = tf.layers.Dense(n)(dec)
            dec = tf.layers.LeakyReLU(self.alpha)(dec)
            dec = tf.layers.BatchNormalization()(dec)
            n = n * 2
        output_dec = tf.layers.Dense(self.npcs, activation='tanh')(dec)
        generator_decoder = tf.Model(input_dec, output_dec, name='Decoder')

        generator_decoder.summary()
        return generator_decoder


    def train(self, epochs, batch_size=128, sample_interval=50, n_critic=5):

        # Load and pre process the data

        pcs_trun = np.load(self.directory_data + '/' + 'pcs_' + self.field_name + '_' +
                           self.observationPeriod + '.npy')

        np.random.seed(42)

        min_ls = np.min(pcs_trun)
        max_ls = np.max(pcs_trun)
        min = -1
        max = +1

        def scaler(x, xmin, xmax, min, max):
            scale = (max - min) / (xmax - xmin)
            xScaled = scale * x + min - xmin * scale
            return xScaled

        ls_scaled = scaler(pcs_trun, min_ls, max_ls, min, max)

        global X_train, y_train, X_all

        X_all = ls_scaled

        if self.GANorWGAN == 'WGAN':
            real = -np.ones(batch_size)
            fake = np.ones(batch_size)

        if self.GANorWGAN == 'GAN':
            real = np.ones(batch_size)
            fake = np.zeros(batch_size)

        # Training the model
        for epoch in range(epochs):
            c1_tmp, c2_tmp = list(), list()

            # Training the discriminator more often than the generator
            for _ in range(n_critic):
                # Randomly selected samples and noise
                randomIndex = np.random.randint(0, X_all.shape[0], size=batch_size)
                noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
                # Select a random batch of input
                real_seqs = X_all[randomIndex]

                # Generate a batch of new outputs (in the latent space) predicted by the encoder
                gen_seqs = self.generator_encoder.predict(real_seqs)

                # Train the discriminator
                # The arbitrary noise is considered to be a "real" sample
                d_loss_real = self.discriminator.train_on_batch(noise, real)
                c1_tmp.append(d_loss_real)
                # The latent space generated by the encoder is considered a "fake" sample
                d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
                c2_tmp.append(d_loss_fake)

            self.c1_hist.append(np.mean(c1_tmp))
            self.c2_hist.append(np.mean(c2_tmp))

            # Training the stacked model
            g_loss = self.combined.train_on_batch(real_seqs, [real_seqs, real])
            self.g_hist.append(g_loss)
            print("%d [C1 real: %f, C2 fake: %f], [G loss: %f, mse: %f]" % (epoch, self.c1_hist[epoch], self.c2_hist[epoch], g_loss[0], g_loss[1]))

            # Checkpoint progress: Plot losses and predicted data
            if epoch % sample_interval == 0:

                self.plot_loss(epoch)
                self.plot_values(epoch)
                self.generator_encoder.save(self.directory_data + '/' + 'AAE_MV_generator_encoder_Full_WGAN_' +
                                            self.field_name + '_' + str(self.latent_dim) + '_' + str(epoch),
                                            save_format='tf')
                self.generator_decoder.save(self.directory_data + '/' + 'AAE_MV_generator_decoder_Full_WGAN_' +
                                            self.field_name + '_' + str(self.latent_dim) + '_' + str(epoch),
                                            save_format='tf')

                self.discriminator.save(self.directory_data + '/' + 'AAE_MV_discriminator_Full_WGAN_' +
                                        self.field_name + '_' + str(self.latent_dim) + '_' + str(epoch),
                                        save_format='tf')

    # Plots the (W)GAN related losses at every sample interval

    def plot_loss(self, epoch):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.plot(self.c1_hist, c='red')
        plt.plot(self.c2_hist, c='blue')
        plt.plot(self.g_hist, c='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("GAN Loss per Epoch")
        plt.legend(['C real', 'C fake', 'Generator'])

        plt.subplot(1,2,2)
        plt.subplot(self.g_hist, c='green')
        plt.xlabel('Epoch')
        plt.ylabel('Mean squared error')
        plt.title('MSE loss')
        plt.savefig(self.directory_data + '/' + 'Losses_AAE_MV-PCAE_WGAN_' + self.field_name + '_' + '_' + str(epoch) +
                    '_' + str(self.latent_dim) + '.png')
        plt.close()

    # Plots predicted in the first 8 latent dimension at every sample interval

    def plot_values(self, epoch):

        prediction = self.generator_decoder.predict(self.generator_encoder(X_all))
        for i in np.arange(12):
            plt.subplot(3, 4, i+1)
            plt.plot(X_all[:, i])
            plt.plot(prediction[:, i], alpha=0.8)

            #plt.legend(['Prediction', 'GT'])
        plt.tight_layout()
        plt.savefig(self.directory_data + '/' + 'Plots_AAE_MV-PCAE_WGAN_' + self.field_name + '_' + '_' + str(epoch) +
                    '_' + str(self.latent_dim) + '.png')
        plt.close()

if __name__ == '__main__':
    directory_data = '/data/'
    field_name = 'Velocity'

    epochs = 100000
    batch_size = 32
    n_critic = 5
    sample_interval = 10000
    latent_dim = 4
    npcs = 1000
    #Initial number of nodes for the AE
    initNNodes = 64
    #Training method
    GANorWGAN = 'WGAN'
    advAE = AAE(directory_data=directory_data,
              field_name=field_name,
              npcs=npcs,
              initNNodes=initNNodes,
              latent_dim=latent_dim,
              GANorWGAN=GANorWGAN)

    advAE.train(epochs=epochs,
              batch_size=batch_size,
              sample_interval=sample_interval,
              n_critic = n_critic)
