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

        np.savetxt(f"./data/ising_data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat_null)
        np.savetxt(f"./data/ising_data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat_alt)

    weights_dict[sample_size] = sample_size_weight_dict

with open("data/ising_data/weights_dict.p", "wb") as fp:
    pickle.dump(weights_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)


###################
# Use mixed model #
###################
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

for sample_size in hp.sample_size_vet:
    for i in range(hp.simulation_times):
        z_mat, label_vet = make_blobs(n_samples=sample_size, centers=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        permutation_vet = np.random.permutation(len(label_vet))
        z_mat = z_mat[permutation_vet]
        label_vet = label_vet[permutation_vet]

