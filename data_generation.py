import numpy as np
import generate_train_fucntions as gt
import tensorflow as tf
from hyperparameters import hidden_1_out_dim, simulation_times, dim_z, sample_size_vet

###########################
# Simulate under the null #
###########################
# Create null network
tf.random.set_seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape=(dim_z, hidden_1_out_dim), mean=1, stddev=1)
linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = tf.random.normal(shape=(hidden_1_out_dim, 2), mean=1, stddev=1)
linear_2_bias_array = tf.zeros(shape=(2,))

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])


###################
# Data generation #
###################
np.random.seed(1)
for sample_size in sample_size_vet:
    z_mat = tf.random.normal(mean=0, stddev=10, shape=(sample_size, dim_z))
    np.savetxt("./data/z_%d.txt" % sample_size, z_mat)

    for i in range(simulation_times):
        ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=null_network_generate, null_boolean=True,
                                              hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                              batch_size=50, epoch=1000)
        x_y_mat = ising_simulation.generate_x_y_mat()
        np.savetxt(f"./data/x_y_mat_{sample_size}_{i}.txt", x_y_mat)

i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])