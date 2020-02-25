import numpy as np
import os

# Seed for random rate.
seed_index=1

###################
# Data generation #
###################
# Number of trails we simulate for each sample size
number_of_trails = 1000

# Sample size we simulate.
sample_size_vet = np.array([30, 100, 500, 1000])

# Number of latent groups within samples.
latent_group_number = 4

# The radius we use for dividing z in to two groups under the mixture data scenario.
null_cut_off_radius = 1
alt_cut_off_radius = 1.539

##################
# Nerual network #
##################
# Dimension of the hidden layer in the true network.
hidden_1_out_dim = 3

# Dimension of the random_variables we condition on.
dim_z = 3

# Training epochs for samples sizes in the sample_size_vet
epoch_vet = np.array([250, 250, 100, 90])

# buffer size for Tensorflow dataset.
buffer_size = 1024

# batch size for training.
batch_size = 100

# learning rate for gradient descent.
learning_rate = 0.005

###############################
# Misspecified neural network #
###############################
wrong_number_forward_elu_layer = 2
wrong_hidden_dim=2


####################################
# Architecture on the mixture data #
####################################
mixture_number_forward_elu_layer = 1
mixture_hidden_dim = 12


############
# Training #
############
number_of_test_samples_vet = [5, 10, 50, 100]


################
# Multiprocess #
################
# Number of process Pool function will run in parallel.
process_number = os.cpu_count()


#########################
# Tuning Ising network. #
#########################



