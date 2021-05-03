from pathlib import Path
import numpy as np
import generate_train_functions as gt
import tensorflow as tf
import pickle
from sklearn.datasets import make_blobs
import hyperparameters as hp

####################
# Create Directory #
####################
for data in ["ising_data", "mixture_data"]:
    Path(f"./data/{data}/z_mat").mkdir( parents=True, exist_ok=True)
    for hypothesis in ["null", "alt"]:
        Path(f"./data/{data}/{hypothesis}").mkdir(parents=True, exist_ok=True)


seed_index = 1

##########################
# Use the mixture model. #
##########################
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

null_cut_off_radius = hp.null_cut_off_radius
alt_cut_off_radius = hp.alt_cut_off_radius

for sample_size in hp.sample_size_vet:
    for trial_index in np.arange(hp.number_of_trials):
        z_mat = tf.random.normal(mean=0, stddev=1, shape=(sample_size, hp.dim_z))

        # H0
        null_p_mat = gt.conditional_pmf_collection_mixture(z_mat=z_mat, is_null_boolean=True,
                                                           cut_off_radius=hp.null_cut_off_radius)
        null_x_y_mat = gt.generate_x_y_mat(null_p_mat)

        # H1
        alt_p_mat = gt.conditional_pmf_collection_mixture(z_mat=z_mat, is_null_boolean=False,
                                                          cut_off_radius=hp.alt_cut_off_radius)
        alt_x_y_mat = gt.generate_x_y_mat(alt_p_mat)

        np.savetxt(f"./data/mixture_data/z_mat/z_mat_{sample_size}_{trial_index}.txt", z_mat)
        np.savetxt(f"./data/mixture_data/null/x_y_mat_{sample_size}_{trial_index}.txt", null_x_y_mat)
        np.savetxt(f"./data/mixture_data/alt/x_y_mat_{sample_size}_{trial_index}.txt", alt_x_y_mat)


########################
# Use the Ising model. #
########################
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

weights_dict = dict()
for sample_size in hp.sample_size_vet:
    sample_size_weight_dict = dict()
    for i in range(hp.number_of_trials):
        # Generate z_mat
        centers = [np.repeat(0.5, 3), np.repeat(-0.5, 3)]
        z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
        np.savetxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{i}.txt", z_mat)

        null_network_generate, alt_network_generate, weights_list = \
            gt.data_generate_network(weights_distribution_string="glorot_normal")
        sample_size_weight_dict[i] = weights_list

        x_y_mat_null = gt.generate_x_y_mat_ising(ising_network=null_network_generate, z_mat=z_mat)
        x_y_mat_alt = gt.generate_x_y_mat_ising(ising_network=alt_network_generate, z_mat=z_mat)

        np.savetxt(f"./data/ising_data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat_null)
        np.savetxt(f"./data/ising_data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat_alt)

    weights_dict[sample_size] = sample_size_weight_dict

with open("data/ising_data/weights_dict.p", "wb") as fp:
    pickle.dump(weights_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

