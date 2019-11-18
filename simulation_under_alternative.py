import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import matplotlib.pyplot as plt
import pickle
from generate_z import dim_z, sample_size_vet

####################
# Hyper parameters #
####################
hidden_1_out_dim = 3
# Number of times we run the simulation for each sample size
simulation_times = 1000
epoch_vet = [250, 250, 100, 90]

##################################
# Simulate under the alternative #
##################################
# Create null network
tf.random.set_seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 3)
null_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape=(dim_z, hidden_1_out_dim), mean=1, stddev=1)
linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = tf.random.normal(shape=(hidden_1_out_dim, 3), mean=1, stddev=1)
linear_2_bias_array = tf.zeros(shape=(3,))

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])