import numpy as np
import generate_train_fucntions as gt
import tensorflow as tf
import pickle
from sklearn.datasets.samples_generator import make_blobs
import hyperparameters as hp

seed_index = 1
tf.random.set_seed(seed_index)
np.random.seed(seed_index)


##########################
# Use the mixture model. #
##########################
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

for sample_size in hp.sample_size_vet:
    for i in range(hp.number_of_trails):
        x_y_mat_null = 2 * np.random.binomial(n=1, p=0.5, size=(sample_size, 2)) - 1

        z_mat, label_vet = make_blobs(n_samples=sample_size, centers=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        x_y_mat_alt, p_mat_alt = gt.mixture_model_x_y_mat(label_vet=label_vet)

        np.savetxt(f"./data/mixture_data/z_mat/z_mat_{sample_size}_{i}.txt", z_mat)
        np.savetxt(f"./data/mixture_data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat_null)
        np.savetxt(f"./data/mixture_data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat_alt)
        np.savetxt(f"./data/mixture_data/alt/p_mat_alt_{sample_size}_{i}.txt", p_mat_alt)



########################
# Use the Ising model. #
########################
tf.random.set_seed(seed_index)
np.random.seed(seed_index)
weights_dict = dict()
for sample_size in hp.sample_size_vet:
    sample_size_weight_dict = dict()
    for i in range(hp.number_of_trails):
        # Generate z_mat
        centers = [np.repeat(0.5, 3), np.repeat(-0.5, 3)]
        z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
        # z_mat = tf.random.normal(mean=0, stddev=10, shape=(sample_size, dim_z))
        np.savetxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{i}.txt", z_mat)

        null_network_generate, alt_network_generate, weights_list = gt.data_generate_network()
        sample_size_weight_dict[i] = weights_list

        x_y_mat_null = gt.generate_x_y_mat_ising(ising_network=null_network_generate, z_mat=z_mat, null_boolean=True,
                                           sample_size=sample_size)
        x_y_mat_alt = gt.generate_x_y_mat_ising(ising_network=alt_network_generate, z_mat=z_mat, null_boolean=False,
                                          sample_size=sample_size)

        np.savetxt(f"./data/ising_data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat_null)
        np.savetxt(f"./data/ising_data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat_alt)

    weights_dict[sample_size] = sample_size_weight_dict

with open("data/ising_data/weights_dict.p", "wb") as fp:
    pickle.dump(weights_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)


