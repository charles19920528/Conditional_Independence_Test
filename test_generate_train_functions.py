import generate_train_functions_nightly as gt
import tensorflow as tf
import numpy as np

tf.random.set_seed(0)
np.random.seed(0)


# Test the IsingNetwork class.
null_ising_network = gt.IsingNetwork(input_dim=3, hidden_1_out_dim=3, output_dim=2)
null_ising_network.dummy_run()
alt_ising_network = gt.IsingNetwork(input_dim=3, hidden_1_out_dim=3, output_dim=3)
alt_ising_network.dummy_run()


# Test pmf_collection function.
null_parameter_mat = tf.constant([[-1, 0], [1, 2], [-3, -2]])
null_pmf_collection_mat = gt.pmf_collection(parameter_mat=null_parameter_mat)

alt_parameter_mat = tf.constant([[-1, 0, 1], [1, 2, 3], [-3, 1, -2]])
alt_pmf_collection_mat = gt.pmf_collection(parameter_mat=alt_parameter_mat)


# Test log_ising_likelihood function.
x_y_mat = tf.constant([[1, 1], [1, -1], [-1, -1]])
null_log_likelihood = gt.log_ising_likelihood(x_y_mat = x_y_mat, parameter_mat = null_parameter_mat)
alt_log_likelihood = gt.log_ising_likelihood(x_y_mat = x_y_mat, parameter_mat = alt_parameter_mat)


# Test generate_x_y_mat function.
p_mat = tf.constant([[0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], [0, 1, 0, 0]])
x_y_mat = gt.generate_x_y_mat(p_mat)


# Test generate_x_y_mat_ising function.
z_mat = tf.random.normal(shape=(4, 3))
null_x_y_mat = gt.generate_x_y_mat_ising(ising_network=null_ising_network, z_mat=z_mat)
alt_x_y_mat = gt.generate_x_y_mat_ising(ising_network=alt_ising_network, z_mat=z_mat)


# Test data_generate_network function.
null_network_generate, alt_network_generate, weights_list = \
    gt.data_generate_network(weights_distribution_string="glorot_normal")
null_network_generate(z_mat)
null_network_generate.get_weights()

