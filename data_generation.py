import numpy as np
import generate_train_functions as gt
import tensorflow as tf
import pickle
from sklearn.datasets import make_blobs
import multiprocessing as mp
from functools import partial
import hyperparameters as hp

import pathlib
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#############################
# Helper functions for pool #
#############################
def _ising_generate_one_trial(trial_index, sample_size, sample_size_weight_dict):
    """
    Generate data under the Ising model; save the x_y_mat and z_mat a; and update the sample_size_weight_dict.

    :param trial_index: A non-negative integer.
    :param sample_size: An integer.
    :param sample_size_weight_dict: A dictionary.

    :return:
        None
    """
    centers = [np.repeat(0.5, 3), np.repeat(-0.5, 3)]
    z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
    np.savetxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{trial_index}.txt", z_mat)

    null_network_generate, alt_network_generate, weights_list = \
        gt.data_generate_network(weights_distribution_string="glorot_normal")

    x_y_mat_null = gt.generate_x_y_mat_ising(ising_network=null_network_generate, z_mat=z_mat)
    x_y_mat_alt = gt.generate_x_y_mat_ising(ising_network=alt_network_generate, z_mat=z_mat)

    np.savetxt(f"./data/ising_data/null/x_y_mat_{sample_size}_{trial_index}.txt", x_y_mat_null)
    np.savetxt(f"./data/ising_data/alt/x_y_mat_{sample_size}_{trial_index}.txt", x_y_mat_alt)

    return trial_index, weights_list


def _mixture_generate_one_trial(trial_index, sample_size):
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


#############################
# Create paths if necessary #
#############################
path_list = ["./data/mixture_data/z_mat", "./data/mixture_data/null", "./data/mixture_data/alt",
             "./data/ising_data/z_mat", "./data/ising_data/null", "./data/ising_data/alt"]
for path_name in path_list:
    path = pathlib.Path(path_name)
    path.mkdir(parents=True, exist_ok=True)

seed_index = 1

pool = mp.Pool(processes=hp.process_number)
trial_index_vet = np.arange(hp.number_of_trials)

##########################
# Use the mixture model. #
##########################
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

null_cut_off_radius = hp.null_cut_off_radius
alt_cut_off_radius = hp.alt_cut_off_radius

for sample_size in hp.sample_size_vet:
    pool.map(partial(_mixture_generate_one_trial, sample_size=sample_size), trial_index_vet)

    print(f"Mixture data. Sample size {sample_size} Done")

########################
# Use the Ising model. #
########################
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

weights_dict = dict()
for sample_size in hp.sample_size_vet:
    sample_size_weight_dict = dict()
    trial_weights_vet = pool.map(partial(_ising_generate_one_trial, sample_size=sample_size,
                                         sample_size_weight_dict=sample_size_weight_dict), trial_index_vet)
    # for i in range(hp.number_of_trials):
    # Generate z_mat
    # centers = [np.repeat(0.5, 3), np.repeat(-0.5, 3)]
    # z_mat = make_blobs(n_samples=sample_size, centers=centers)[0]
    # np.savetxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{i}.txt", z_mat)
    #
    # null_network_generate, alt_network_generate, weights_list = \
    #     gt.data_generate_network(weights_distribution_string="glorot_normal")
    # sample_size_weight_dict[i] = weights_list
    #
    # x_y_mat_null = gt.generate_x_y_mat_ising(ising_network=null_network_generate, z_mat=z_mat)
    # x_y_mat_alt = gt.generate_x_y_mat_ising(ising_network=alt_network_generate, z_mat=z_mat)
    #
    # np.savetxt(f"./data/ising_data/null/x_y_mat_{sample_size}_{i}.txt", x_y_mat_null)
    # np.savetxt(f"./data/ising_data/alt/x_y_mat_{sample_size}_{i}.txt", x_y_mat_alt)

    weights_dict[sample_size] = dict(trial_weights_vet)

    print(f"Ising data. Sample size {sample_size} Done")

with open(f"data/ising_data/weights_dict_{hp.sample_size_vet[-1]}.p", "wb") as fp:
    pickle.dump(weights_dict, fp, protocol=4)

pool.close()
pool.join()
