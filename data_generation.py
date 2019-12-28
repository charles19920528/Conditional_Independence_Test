import numpy as np
import generate_train_fucntions as gt
import tensorflow as tf
from sklearn.datasets.samples_generator import make_blobs
from hyperparameters import hidden_1_out_dim, simulation_times, dim_z, sample_size_vet, cluster_number

seed_index = 1
tf.random.set_seed(seed_index)
np.random.seed(seed_index)


# Create the alternative network
alt_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 3)
alt_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape = (dim_z, hidden_1_out_dim), mean = 0, stddev = 1)
linear_1_bias_array = tf.zeros(shape = (hidden_1_out_dim, ))

linear_2_weight_array = tf.random.normal(shape = (hidden_1_out_dim, 3), mean = 0, stddev = 1)
linear_2_bias_array = tf.zeros(shape = (3,))

alt_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])

# Create the null network
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

null_linear_2_weight_array = linear_2_weight_array[:, :2]
null_linear_2_bias_array = linear_2_bias_array[ : 2]

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   null_linear_2_weight_array, null_linear_2_bias_array])

# Check signal strength using KL divergence
z_mat_example = tf.random.normal(mean = 0, stddev = 5, shape = (30, dim_z))
null_parameter = null_network_generate(z_mat_example)
alt_parameter = alt_network_generate(z_mat_example)

# A little sanity check
np.sum(null_parameter == alt_parameter[:, :2]) == (null_parameter.shape[0] * null_parameter.shape[1])

# Average kl divergence. It seems to be greatly affected by the magnitude of the weights.
kl_divergence = gt.kl_divergence(null_parameter, alt_parameter)


##################################
# Simulate under the alternative #
##################################
ising_generate_instance = gt.IsingGenerate(ising_network = alt_network_generate, dim_z = dim_z,
                                           null_boolean = False, sample_size)



































# Generate data
for sample_size in sample_size_vet:
#    centers = tf.random.normal((cluster_number, dim_z), mean=0, stddev=10)
#    z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
    z_mat = tf.random.normal(mean=0, stddev=10, shape=(sample_size, dim_z))
    np.savetxt("./data/alt/z_mat_%d.txt" % sample_size, z_mat)

    ising_simulation_alt = gt.IsingSimulation(z_mat=z_mat, true_network=alt_network_generate, null_boolean=False,
                                              hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                              batch_size=50, epoch=1000)
    for i in range(simulation_times):
        x_y_mat = ising_simulation_alt.generate_x_y_mat()
        np.savetxt(f"./data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat)





###########################
# Simulate under the null #
###########################
# Create null network
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape=(dim_z, hidden_1_out_dim), mean=1, stddev=1)
linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = tf.random.normal(shape=(hidden_1_out_dim, 2), mean=1, stddev=1)
linear_2_bias_array = tf.zeros(shape=(2,))

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])


# Generate data
for sample_size in sample_size_vet:
    z_mat = tf.random.normal(mean=0, stddev=10, shape=(sample_size, dim_z))
#    centers = tf.random.normal((cluster_number, dim_z), mean=0, stddev=10)
#    z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
    np.savetxt("./data/null/z_mat_%d.txt" % sample_size, z_mat)

    ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=null_network_generate, null_boolean=True,
                                              hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                              batch_size=50, epoch=1000)
    for i in range(simulation_times):
        x_y_mat = ising_simulation.generate_x_y_mat()
        np.savetxt(f"./data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat)








