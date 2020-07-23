import time
import multiprocessing as mp
import os
import numpy as np
import simulation_functions as sf
import generate_train_functions as gt
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

pool = mp.Pool(processes=hp.process_number)

# Set up
mixture_result_dict_name = f"mixture_data_{hp.mixture_number_forward_layer}_{hp.mixture_hidden_dim}"
mixture_network_model_class_kwargs = {"number_forward_layers": hp.mixture_number_forward_layer,
                                      "input_dim": hp.dim_z, "hidden_dim": hp.mixture_hidden_dim, "output_dim": 3}
mixture_network_model_class_kwargs_vet = [mixture_network_model_class_kwargs for _ in range(len(hp.sample_size_vet))]

true_result_dict_name = f"ising_data_true_architecture"
true_network_model_class_kwargs = {"number_forward_layers": 1, "input_dim": hp.dim_z,
                                   "hidden_dim": hp.hidden_1_out_dim, "output_dim": 3}
true_network_model_class_kwargs_vet = [true_network_model_class_kwargs for _ in range(len(hp.sample_size_vet))]

trial_index_vet = np.arange(2)
# Ising
# Null data
start_time = time.time()
sf.ising_bootstrap_loop(pool=pool, data_directory_name="ising_data", scenario="null",
                        ising_simulation_result_dict_name=true_result_dict_name, result_dict_name="bootstrap",
                        trial_index_vet=trial_index_vet, network_model_class=gt.FullyConnectedNetwork,
                        network_model_class_kwargs_vet=true_network_model_class_kwargs_vet,
                        number_of_bootstrap_samples=2, epoch_vet=hp.ising_epoch_vet,
                        sample_size_vet=[50, 100])


print(f"Bootstrap simulation under null alt data takes {time.time() - start_time} seconds to finish.")

"""
Not in use
####################################
# Simulate argmax Gaussian Process #
####################################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

# Ising
# Null data
start_time = time.time()
sf.argmax_simulation_loop(pool=pool, trial_index_vet=np.arange(hp.number_of_trials), sample_size_vet=hp.sample_size_vet,
                          scenario="null", data_directory_name="ising_data",
                          ising_simulation_result_dict_name=true_result_dict_name,
                          network_model_class=gt.FullyConnectedNetwork,
                          network_model_class_kwargs_vet=true_network_model_class_kwargs_vet,
                          network_net_size=hp.network_net_size, number_of_nets=hp.number_of_nets,
                          result_dict_name="gaussian_process")

print(f"Gaussian process simulation under null Ising data takes {time.time() - start_time} seconds to finish.")


# Alt
start_time = time.time()
sf.argmax_simulation_loop(pool=pool, trial_index_vet=np.arange(hp.number_of_trials), sample_size_vet=[1000],
                          scenario="alt", data_directory_name="ising_data",
                          ising_simulation_result_dict_name=true_result_dict_name,
                          network_model_class=gt.FullyConnectedNetwork,
                          network_model_class_kwargs_vet=true_network_model_class_kwargs_vet,
                          network_net_size=hp.network_net_size, number_of_nets=hp.number_of_nets,
                          result_dict_name="gaussian_process")

print(f"Gaussian process simulation under alternative Ising data takes {time.time() - start_time} seconds to finish.")
"""