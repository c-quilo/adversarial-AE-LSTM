#######################################
# Hyperparameters #
#######################################

#Number of epochs for training
epochs = 50001

#Size of batch for training
batch_size = 32

#Number of updates of the discriminator before training the generator
n_critic = 5

#Number of time-steps to look back when training LSTM
look_back = 5

#Checkpoint (number of epochs) for saving models
sample_interval = 10000
