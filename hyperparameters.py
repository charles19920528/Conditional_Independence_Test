import numpy as np
import os

# Seed for random rate.
seed_index=1

###################
# Data generation #
###################
# Dimension of the random_variables we condition on.
dim_z = 3

# Number of trials we simulate for each sample size
number_of_trials = 1000

# Sample size we simulate.
sample_size_vet = np.array([100, 500, 1000])

# The radius we use for dividing z in to two groups under the mixture data scenario.
# When dim_z = 3, the null_cut_off_radius will make P(norm(z) < alt_cut_off_radius) is roughly 0.65.
# When dim_z = 3, the alt_cut_off_radius will make P(norm(z) < alt_cut_off_radius) is roughly 0.5.
null_cut_off_radius = 1.046 * np.sqrt(dim_z)
alt_cut_off_radius = 0.8875 * np.sqrt(dim_z)

p_null_norm_less = 0.65
p_g = 0.4
p_l = 0.8

##################
# Nerual network #
##################
# Dimension of the hidden layer in the true network.
hidden_1_out_dim = 90

# Training epochs for samples sizes in the sample_size_vet
ising_epoch_vet = np.array([1, 8, 11])

full_model_mixture_epoch_vet_alt = np.array([30, 110, 74])
full_model_mixture_epoch_vet_null = np.array([42, 76, 60])

# buffer size for Tensorflow dataset.
buffer_size = 1024

# batch size for training.
batch_size = 100

# learning rate for gradient descent.
learning_rate = 0.01
learning_rate_mixture = 0.01


####################################
# Architecture on the mixture data #
####################################
mixture_number_forward_layer_null = 16
mixture_hidden_dim_null = 40

mixture_number_forward_layer_alt = 16
mixture_hidden_dim_alt = 40


############
# Training #
############
test_sample_prop = 0.1
# test_prop_list = [0.01, 0.05, 0.1, 0.4]
test_prop_list = [0, 0.1]

################
# Multiprocess #
################
# Number of process Pool function will run in parallel.
process_number = os.cpu_count()


###########################################
# Sampling distribution of test statistic #
###########################################
number_of_boostrap_samples = 10**3

