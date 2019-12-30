##################
# Nerual network #
##################
# Dimension of the hidden layer in the true network.
hidden_1_out_dim = 3

# Number of times we run the simulation for each sample size
simulation_times = 1000

# Dimension of the random_variables we condition on.
dim_z = 3

# Sample size we simulate.
sample_size_vet = [30, 100, 500, 1000]

# Training epochs for samples sizes in the sample_size_vet
epoch_vet = [250, 250, 100, 90]

####################
# Chi squared test #
####################
# Number of process Pool function will run in parallel.
process_number = 10


