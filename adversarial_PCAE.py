from __future__ import print_function, division

import tensorflow.keras as tf
import tensorflow
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as backend
import matplotlib.pyplot as plt
import numpy as np
from keras import backend
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
class GAN():
    # Implementation of wasserstein loss
    def __init__(self, directory_data, field_name, npcs, latent_dim, look_back, GANorWGAN):
        def wasserstein_loss(y_true, y_pred):
            return backend.mean(y_true * y_pred)

        self.field_name = field_name
        self.directory_data = directory_data
        self.latent_dim = latent_dim
        self.npcs = npcs
        self.look_back = look_back
        self.constraint = 0.01
        self.dropoutNumber = 0.5
        self.alpha = 0.3
        self.hiddenNodes = 64
        self.look_back = look_back
        self.GANorWGAN = GANorWGAN

        self.c1_hist = []
        self.c2_hist = []
        self.g_hist = []

        self.optimizer = tf.optimizers.RMSprop(lr=0.00005)

        if GANorWGAN == 'WGAN':
            self.loss = wasserstein_loss
        elif GANorWGAN == 'GAN'
            self.loss = 'binary_crossentropy'

        # Adversarial autoencoder loss
        self.loss_gen = 'mse'
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the adversarial LSTM
        self.alstm = self.build_alstm()

        real_input = tf.Input(shape=(1, self.latent_dim))
        alstm_output = self.alstm(real_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        discriminator_output = self.discriminator(alstm_output)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = tf.Model(real_input, [alstm_output, discriminator_output], name='ALSTM')
        self.combined.compile(loss=[self.loss_gen, self.loss], loss_weights=[0.999, 0.001], optimizer=self.optimizer)

    def build_discriminator(self):
        init = RandomNormal(stddev=0.02)
        const = ClipConstraint(0.01)

        # Discriminator is a bidirectional LSTM with input latent dimensions and output dim = 1
        in_disc = tf.Input(shape=(1, self.latent_dim))
        lstm_1 = tf.layers.Bidirectional(tf.layers.LSTM(self.hiddenNodes, return_sequences=False))(in_disc)
        dp_1 = tf.layers.Dropout(self.dropoutNumber)(lstm_1)
        bn_1 = tf.layers.BatchNormalization()(dp_1)
        rv_1 = tf.layers.RepeatVector(1)(bn_1)

        # Discriminator output
        disc_output = tf.layers.TimeDistributed(tf.layers.Dense(1, activation='linear'))(rv_1)

        discriminator = tf.Model(in_disc, disc_output)
        discriminator.compile(loss=self.loss, optimizer=self.optimizer)

        return discriminator

    def build_alstm(self):
        init = RandomNormal(stddev=0.02)
        init = tf.initializers.RandomNormal(stddev=0.02)

        # BiLSTM generator
        in_gen = tf.Input(shape=(self.look_back, self.latent_dim))

        lstm_1 = tf.layers.Bidirectional(tf.layers.LSTM(self.hiddenNodes, return_sequences=False))(in_gen)
        dp_1 = tf.layers.Dropout(self.dropoutNumber)(lstm_1)
        bn_1 = tf.layers.BatchNormalization()(dp_1)
        rv_1 = tf.layers.RepeatVector(1)(bn_1)
        # BiLSTM generator output
        gen_output = tf.layers.TimeDistributed(tf.layers.Dense(self.latent_dim, activation='linear'))(rv_1)

        alstm = tf.Model(in_gen, gen_output)
        return alstm

    def train(self, epochs, batch_size=128, sample_interval=50, n_critic=5):

        # Load principal components and use previously trained Adversarial encoder to obtain latent values

        pcs_trun = np.load(self.directory_data + '/' + 'pcs_' + self.field_name + '_' +
                           self.observationPeriod + '.npy')
        # Load encoder
        generator_enc = tf.models.load_model(self.directory_data + '/' + 'AAE_MV_generator_encoder_Full_'
                                             + self.field_name + '_' + str(latent_dim) + '_' + str(100000))

        np.random.seed(42)

        # Calculate maximum and minimum to normalize between -1 and +1
        min_ls = np.min(pcs_trun)
        max_ls = np.max(pcs_trun)
        min = -1
        max = +1

        global X_all, y_all, X_train, X_test, y_train, y_test
        def scaler(x, xmin, xmax, min, max):
            scale = (max - min) / (xmax - xmin)
            xScaled = scale * x + min - xmin * scale
            return xScaled

        def lookBack(X, look_back=1):
            # look_back = 10
            X_lb = np.empty((X.shape[0] - look_back + 1, look_back, X.shape[1]))
            # X_test = np.empty((look_back, X_test.shape[1], X_train.shape[0] - look_back + 1))
            ini = 0
            fin = look_back
            for i in range(X.shape[0] - look_back + 1):
                X_lb[i, :, :] = (X[ini:fin, :])
                ini = ini + 1
                fin = fin + 1
            return X_lb

        # Normalise the values between -1 and +1
        ls_scaled = scaler(pcs_trun, min_ls, max_ls, min, max)

        # Encode the normalised values
        latent_values = generator_enc.predict(ls_scaled)

        lv_scaled = latent_values

        # Training data
        # All data is used for training, testing is performed in unseen data
        X_all = lookBack(lv_scaled[:-1, :], look_back)
        # Delta latent space with respect to time
        y_all = lv_scaled[look_back:]-lv_scaled[look_back-1:-1]
        y_all = np.expand_dims(y_all, 1)

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
                noise = np.random.normal(0, 1, size=(batch_size, self.look_back, self.latent_dim))

                # Select a random batch for input
                real_input = X_all[randomIndex]
                real_output = y_all[randomIndex]

                # Generate a batch of new outputs (in the latent space) predicted by the generator
                alstm_seqs = self.alstm.predict(noise)

                # Train the discriminator
                # The arbitrary noise is considered to be a "real" sample
                d_loss_real = self.discriminator.train_on_batch(real_output, real)
                c1_tmp.append(d_loss_real)
                # The latent space generated by the encoder is considered a "fake" sample
                d_loss_fake = self.discriminator.train_on_batch(alstm_seqs, fake)
                c2_tmp.append(d_loss_fake)
                #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            self.c1_hist.append(np.mean(c1_tmp))
            self.c2_hist.append(np.mean(c2_tmp))

            #  Training the generator through the stacked model
            g_loss = self.combined.train_on_batch(real_input, [real_output, real])
            self.g_hist.append(g_loss)
            print("%d [C1 real: %f, C2 fake: %f], [G loss: %f, mse: %f]" % (epoch, self.c1_hist[epoch],
                                                                            self.c2_hist[epoch],
                                                                            g_loss[0],
                                                                            g_loss[1]))

            # Checkpoint progress: Plot losses and predicted data
            if epoch % sample_interval == 0:

                self.plot_loss(epoch)
                self.plot_values(epoch)
                # Saving the adversarial LSTM
                self.alstm.save(self.directory_data + '/' + 'AAE_generator_LSTM_GAN_noise_'
                                + self.field_name + '_' + str(self.latent_dim) + '_' + str(epoch), save_format='tf')

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
        plt.plot(self.g_hist, c='green')
        plt.xlabel('Epoch')
        plt.ylabel('Mean squared error (-)')
        plt.title('MSE loss')
        plt.legend(['MSE loss'])
        plt.savefig(self.directory_data + '/' + 'Losses_AAE-PCAE_LSTM_GAN_noise_' +
                    self.field_name + '_' + '_' + str(epoch)+ '_' + str(self.latent_dim) +
                    '.png')
        plt.close()

    # Plots predicted in the first 8 latent dimension at every sample interval
    def plot_values(self, epoch):
        prediction = self.alstm.predict(X_all)
        for i in np.arange(8):
            plt.subplot(2, 4, i+1)
            plt.plot(y_all[:, i])
            plt.plot(prediction[:, :, i], alpha=0.8)
            plt.title('LS dim: ' + str(i))
        plt.tight_layout()

        plt.tight_layout()
        plt.savefig(self.directory_data + '/' + 'Plots_AAE-PCAE_LSTM_GAN_noise_' +
                    self.field_name + '_' + '_' + str(epoch) + '_' + str(self.latent_dim) +
                    '.png')
        plt.close()

if __name__ == '__main__':
    directory_data = '/data/'
    field_name = 'Velocity'
    npcs = 1000
    latent_dim = 8
    start_interv = 150
    end_interv = 1150

    latentSpaceDimensions = [8]  # , 16, 32, 64, 128]
    epochs = 100001
    batch_size = 32
    n_critic = 5
    sample_interval = 10000
    look_back = 5

    #Training method
    GANorWGAN = 'WGAN'

    gan = GAN(directory_data=directory_data,
              field_name=field_name,
              npcs=npcs,
              latent_dim = latent_dim,
              look_back=look_back,
              GANorWGAN=GANorWGAN)

    gan.train(epochs=epochs,
              batch_size=batch_size,
              sample_interval=sample_interval,
              n_critic = n_critic)
