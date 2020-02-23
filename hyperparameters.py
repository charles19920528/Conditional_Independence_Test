import numpy as np
import os

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
epoch_vet_misspecified = np.array([250, 250, 70, 70])
hidden_1_out_dim_misspecified = 2
hidden_2_out_dim_misspecified = 2


####################################
# Architecture on the mixture data #
####################################
mixture_hidden_dim_vet = np.array([3, 3, 3, 3])

epoch_vet_mixture_alt = np.array([20, 45, 25, 22])
epoch_vet_mixture_null = np.array([40, 40, 15, 13])
################
# Multiprocess #
################
# Number of process Pool function will run in parallel.
process_number = os.cpu_count()


#########################
# Tuning Ising network. #
#########################



