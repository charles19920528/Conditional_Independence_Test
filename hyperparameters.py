###################
# Data generation #
###################
# Number of trails we simulate for each sample size
number_of_trails = 1000

# Sample size we simulate.
sample_size_vet = [30, 100, 500, 1000]

# Number of latent groups within samples.
latent_group_number = 4

##################
# Nerual network #
##################
# Dimension of the hidden layer in the true network.
hidden_1_out_dim = 3

# Dimension of the random_variables we condition on.
dim_z = 3

# Training epochs for samples sizes in the sample_size_vet
epoch_vet = [250, 250, 100, 90]

# buffer size for Tensorflow dataset.
buffer_size = 1024

# batch size for training.
batch_size = 30

# learning rate for gradient descent.
learning_rate = 0.005

###############################
# Misspecified neural network #
###############################
epoch_vet_misspecified = [250, 250, 70, 70]
hidden_1_out_dim_misspecified = 2
hidden_2_out_dim_misspecified = 2

################
# Multiprocess #
################
# Number of process Pool function will run in parallel.
process_number = 22


