import numpy as np
import generate_train_fucntions as gt
import tensorflow as tf
from sklearn.datasets.samples_generator import make_blobs
from hyperparameters import hidden_1_out_dim, simulation_times, dim_z, sample_size_vet, cluster_number

seed_index = 1
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

scenario = "null"
sample_size = 30

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
true_network = null_network_generate


centers = tf.random.normal((cluster_number, dim_z), mean=0, stddev=10)
z_mat = np.loadtxt(f"./data/{scenario}/z_mat_{sample_size}.txt")

true_parameter_mat = true_network(z_mat)
p_equal_1_mat_1 = gt.pmf_null(1, true_parameter_mat)


centers = tf.random.normal((cluster_number, dim_z), mean=0, stddev=10)
z_mat = make_blobs(n_samples=100, centers=centers)[0]

true_parameter_mat = true_network(z_mat)
p_equal_1_mat_2 = gt.pmf_null(1, true_parameter_mat)


x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_0.txt")
x_y_mat_1 = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_10.txt")

x_y_mat == x_y_mat_1