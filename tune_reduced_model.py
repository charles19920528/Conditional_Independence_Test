import numpy as np
import tensorflow as tf
import ising_tuning_functions as it
import generate_train_functions as gt
import hyperparameters as hp
import os
import pickle
import multiprocessing as mp


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

trial_index_vet = np.arange(120)
sample_size_vet = hp.sample_size_vet

if len(trial_index_vet) < hp.process_number:
    process_number = len(trial_index_vet)
else:
    process_number = hp.process_number

# Tune model
number_forward_layers = 1
hidden_dim_mixture = 10**7

null_network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                                     "hidden_dim": hidden_dim_mixture, "output_dim": 2}
alt_network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                                     "hidden_dim": hidden_dim_mixture, "output_dim": 3}

np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

null_tuning_result = it.ising_tuning_one_trial(trial_index=0, sample_size=500, scenario="null",
                                               data_directory_name="mixture_data", epoch=50, number_of_test_samples=50,
                                               network_model_class=gt.FullyConnectedNetwork,
                                               network_model_class_kwargs=null_network_kwargs_dict,
                                               learning_rate=10**-6, batch_size=hp.batch_size,
                                               true_weights_dict=None, cut_off_radius=hp.null_cut_off_radius)
# alt_tuning_result = it.ising_tuning_one_trial(trial_index=0, sample_size=500, scenario="null",
#                                                data_directory_name="mixture_data", epoch=100, number_of_test_samples=50,
#                                                network_model_class=gt.FullyConnectedNetwork,
#                                                network_model_class_kwargs=alt_network_kwargs_dict,
#                                                learning_rate=hp.learning_rate,
#                                                true_weights_dict=None, cut_off_radius=hp.null_cut_off_radius)



pool = mp.Pool(processes=process_number)

number_forward_layers = 1
hidden_dim_mixture_vet = [10, 20, 50, 100, 200, 500, 700]
mixture_result_dict_name_vet = [f"mixture_reduced_model_{number_forward_layers}_{hidden_dim}_{hp.learning_rate_mixture}"
                                for hidden_dim in hidden_dim_mixture_vet]



pool.close()
pool.join()