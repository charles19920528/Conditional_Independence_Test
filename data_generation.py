import numpy as np
import generate_train_fucntions as gt
import tensorflow as tf
import pickle
from sklearn.datasets.samples_generator import make_blobs
import hyperparameters as hp

seed_index = 1
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

weights_dict = dict()
for sample_size in hp.sample_size_vet:
    sample_size_weight_dict = dict()
    for i in range(hp.simulation_times):
        # Generate z_mat
        centers = [np.repeat(0.5, 3), np.repeat(-0.5, 3)]
        z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
        # z_mat = tf.random.normal(mean=0, stddev=10, shape=(sample_size, dim_z))
        np.savetxt(f"./data/z_mat/z_mat_{sample_size}_{i}.txt", z_mat)

        null_network_generate, alt_network_generate, weights_list = gt.data_generate_network()
        sample_size_weight_dict[i] = weights_list

        x_y_mat_null = gt.generate_x_y_mat(ising_network=null_network_generate, z_mat=z_mat, null_boolean=True,
                                           sample_size=sample_size)
        x_y_mat_alt = gt.generate_x_y_mat(ising_network=alt_network_generate, z_mat=z_mat, null_boolean=False,
                                          sample_size=sample_size)

        np.savetxt(f"./data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat_null)
        np.savetxt(f"./data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat_alt)

    weights_dict[sample_size] = sample_size_weight_dict

with open("./data/weights_dict.p", "wb") as fp:
    pickle.dump(weights_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)













"""
# Create the alternative network
alt_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 3)
alt_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape=(dim_z, hidden_1_out_dim), mean=0, stddev=1)
linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = tf.random.normal(shape=(hidden_1_out_dim, 3), mean=0, stddev=1)
linear_2_bias_array = tf.zeros(shape=(3,))

alt_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                  linear_2_weight_array, linear_2_bias_array])

# Create the null network
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

null_linear_2_weight_array = linear_2_weight_array[:, :2]
null_linear_2_bias_array = linear_2_bias_array[: 2]

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   null_linear_2_weight_array, null_linear_2_bias_array])
"""


#################
# Generate data #
#################
if __name__ == "__main__":
    for sample_size in sample_size_vet:
        centers = [np.repeat(0.5, 3), np.repeat(-0.5, 3)]
        z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
        # z_mat = tf.random.normal(mean=0, stddev=10, shape=(sample_size, dim_z))

        np.savetxt("./data/z_mat/z_mat_%d.txt" % sample_size, z_mat)

        for i in range(simulation_times):
            x_y_mat_null = gt.generate_x_y_mat(ising_network=null_network_generate, z_mat=z_mat, null_boolean=True,
                                               sample_size=sample_size)
            x_y_mat_alt = gt.generate_x_y_mat(ising_network=alt_network_generate, z_mat=z_mat, null_boolean=False,
                                              sample_size=sample_size)

            np.savetxt(f"./data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat_null)
            np.savetxt(f"./data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat_alt)


# Check signal strength using KL divergence
# z_mat_example = tf.random.normal(mean = 0, stddev = 5, shape = (30, dim_z))
# null_parameter = null_network_generate(z_mat_example)
# alt_parameter = alt_network_generate(z_mat_example)

# A little sanity check
# np.sum(null_parameter == alt_parameter[:, :2]) == (null_parameter.shape[0] * null_parameter.shape[1])

# Average kl divergence. It seems to be greatly affected by the magnitude of the weights.
# kl_divergence = gt.kl_divergence(null_parameter, alt_parameter)
wor