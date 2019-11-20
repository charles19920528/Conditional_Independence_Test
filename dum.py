import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import matplotlib.pyplot as plt
import pickle
from generate_z import dim_z, sample_size_vet
import os

# Only run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


sample_size, epoch = [30, 250]
z_mat = np.loadtxt("./data/z_%d.txt" % sample_size, dtype="float32")

ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=null_network_generate, null_boolean=False,
                                          hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                          batch_size=50, epoch=epoch)
alternative_result_dict_30 = dict()
for j in np.arange(simulation_times):
    x_y_mat = ising_simulation.generate_x_y_mat()
    alternative_result_dict_30[j] = ising_simulation.trainning(x_y_mat, print_loss_boolean=False)
    print("Training finished. Sample size : %d, simulation sample: %d" % (sample_size, j))


with open("./results/alternative_result_dict_30.p", "wb") as fp:
    pickle.dump(alternative_result_dict_30, fp, protocol=pickle.HIGHEST_PROTOCOL)

