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
sample_size_vet = np.array([50, 100, 500, 1000])

# The radius we use for dividing z in to two groups under the mixture data scenario.
# When dim_z = 3, the null_cut_off_radius will make P(x = -y) approximately 0.3252 and p(x = y) approximately 0.675.
# When dim_z = 3, the alt_cut_off_radius will make P(norm(z) < alt_cut_off_radius) roughly 0.5.
null_cut_off_radius = 1.046 * np.sqrt(dim_z)
alt_cut_off_radius = 0.8875 * np.sqrt(dim_z)


##################
# Nerual network #
##################
# Dimension of the hidden layer in the true network.
hidden_1_out_dim = 100

# Training epochs for samples sizes in the sample_size_vet
ising_epoch_vet = np.array([1, 1, 8, 11])
mixture_epoch_vet = np.array([76, 71, 130, 93])
reduced_model_epoch_vet = np.array([8, 8, 14, 21])

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
mixture_number_forward_layer = 1
mixture_hidden_dim = 200


############
# Training #
############
number_of_test_samples_vet = [5, 10, 50, 100]


################
# Multiprocess #
################
# Number of process Pool function will run in parallel.
process_number = os.cpu_count()


###########################################
# Sampling distribution of test statistic #
###########################################
number_of_boostrap_samples = 10**3

