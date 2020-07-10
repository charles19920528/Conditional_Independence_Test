import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import os

import simulation_functions as sf
import generate_train_functions_nightly as gt
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

pool = mp.Pool(processes=hp.process_number)

##########################
# Naive Chi squared test #
##########################
# Ising data
sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="null",
                   data_directory_name="ising_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

# Naive Chisq simulation
sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

###############################
# Stratified Chi squared test #
###############################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

# Ising data
sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="null",
                   data_directory_name="ising_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

# Mixture data
sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials), cluster_number=2)

sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials), cluster_number=2)

###############
# Ising Model #
###############
# Mixture data
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

# Alternative
# Prepare arguments for ising_simulation_loop.
result_dict_name = f"mixture_data_{hp.mixture_number_forward_layer}_{hp.mixture_hidden_dim}"
mixture_network_model_class_kwargs = {"number_forward_layers": hp.mixture_number_forward_layer,
                                      "input_dim": hp.dim_z, "hidden_dim": hp.mixture_hidden_dim, "output_dim": 3}
mixture_network_model_class_kwargs_vet = [mixture_network_model_class_kwargs for i in range(len(hp.sample_size_vet))]

start_time = time.time()
sf.ising_simulation_loop(pool=pool, scenario="alt", data_directory_name="mixture_data",
                         result_dict_name=result_dict_name, trial_index_vet=np.arange(hp.number_of_trials),
                         network_model_class=gt.FullyConnectedNetwork,
                         network_model_class_kwargs_vet=mixture_network_model_class_kwargs_vet,
                         epoch_vet=hp.mixture_epoch_vet, learning_rate=hp.learning_rate_mixture,
                         sample_size_vet=hp.sample_size_vet, number_of_test_samples_vet=hp.number_of_test_samples_vet)

print("Ising simulation under alternative mixture data takes %s seconds to finish." % (time.time() - start_time))


# Null
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

start_time = time.time()
sf.ising_simulation_loop(pool=pool, scenario="null", data_directory_name="mixture_data",
                         result_dict_name=result_dict_name, trial_index_vet=np.arange(hp.number_of_trials),
                         network_model_class=gt.FullyConnectedNetwork,
                         network_model_class_kwargs_vet=mixture_network_model_class_kwargs_vet,
                         epoch_vet=hp.mixture_epoch_vet, learning_rate=hp.learning_rate_mixture,
                         sample_size_vet=hp.sample_size_vet, number_of_test_samples_vet=hp.number_of_test_samples_vet)

print("Ising simulation under null mixture data takes %s seconds to finish." % (time.time() - start_time))


########
# CCIT #
########
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

process_number_ccit = 3
ccit_pool = mp.Pool(processes=process_number_ccit)

# Ising data
start_time = time.time()

sf.simulation_loop(pool=ccit_pool, simulation_wrapper=sf.ccit_wrapper, scenario="null",
                   data_directory_name="ising_data", result_dict_name="ccit")

sf.simulation_loop(pool=ccit_pool, simulation_wrapper=sf.ccit_wrapper, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="ccit")

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))

# mixture data
start_time = time.time()

sf.simulation_loop(pool=ccit_pool, simulation_wrapper=sf.ccit_wrapper, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="ccit")

sf.simulation_loop(pool=ccit_pool, simulation_wrapper=sf.ccit_wrapper, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="ccit")

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))

ccit_pool.close()
ccit_pool.join()

# These simulation used the oracle optimal epoch. They are not in use now.
####################
# True Ising model #
####################
# np.random.seed(hp.seed_index)
# tf.random.set_seed(hp.seed_index)
#
# start_time = time.time()
#
# sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_true_rate_{hp.learning_rate}",
#                                        scenario="null", data_directory_name="ising_data",
#                                        ising_network_class=gt.IsingNetwork,
#                                        input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)
#
# sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_true_rate_{hp.learning_rate}",
#                                        scenario="alt", data_directory_name="ising_data",
#                                        ising_network_class=gt.IsingNetwork,
#                                        input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)
#
# print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))


# Simulate when sample size is 100 using different test sample size.
# for number_of_test_samples in hp.number_of_test_samples_100_vet:
#     epoch_kl_dict_name = f"ising_true_rate_{hp.learning_rate}_n_100_test_{number_of_test_samples}"
#
#     sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=epoch_kl_dict_name, sample_size_vet=[100],
#                                            number_of_test_samples_vet=[number_of_test_samples],
#                                            scenario="null", data_directory_name="ising_data",
#                                            ising_network_class=gt.IsingNetwork,
#                                            input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)
#
#     sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=epoch_kl_dict_name, sample_size_vet=[100],
#                                            number_of_test_samples_vet=[number_of_test_samples],
#                                            scenario="alt", data_directory_name="ising_data",
#                                            ising_network_class=gt.IsingNetwork,
#                                            input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)


#######################
# General Ising Model #
#######################
# Ising data
# np.random.seed(hp.seed_index)
# tf.random.set_seed(hp.seed_index)
#
# start_time = time.time()
#
# sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_wrong_rate_{hp.learning_rate}",
#                                        scenario="null", data_directory_name="ising_data",
#                                        ising_network_class=gt.FullyConnectedNetwork,
#                                        number_forward_elu_layers=hp.wrong_number_forward_elu_layer,
#                                        input_dim=hp.dim_z, hidden_dim=hp.wrong_hidden_dim, output_dim=3)
#
#
# sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_wrong_rate_{hp.learning_rate}",
#                                        scenario="alt", data_directory_name="ising_data",
#                                        ising_network_class=gt.FullyConnectedNetwork,
#                                        number_forward_elu_layers=hp.wrong_number_forward_elu_layer,
#                                        input_dim=hp.dim_z, hidden_dim=hp.wrong_hidden_dim, output_dim=3)
#
# print("Misspecified Ising simulation takes %s seconds to finish." % (time.time() - start_time))


# Mixture data
# epoch_kl_dict_name = f"mixture_{hp.mixture_number_forward_elu_layer}_{hp.mixture_hidden_dim}_{hp.learning_rate_mixture}"
#
# np.random.seed(hp.seed_index)
# tf.random.set_seed(hp.seed_index)
#
# start_time = time.time()
#
# sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=epoch_kl_dict_name, scenario="null",
#                                        data_directory_name="mixture_data", ising_network_class=gt.FullyConnectedNetwork,
#                                        number_forward_elu_layers=hp.mixture_number_forward_elu_layer,
#                                        input_dim=hp.dim_z, hidden_dim=hp.mixture_hidden_dim, output_dim=3,
#                                        learning_rate=hp.learning_rate_mixture)
#
# sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=epoch_kl_dict_name, scenario="alt",
#                                        data_directory_name="mixture_data", ising_network_class=gt.FullyConnectedNetwork,
#                                        number_forward_elu_layers=hp.mixture_number_forward_elu_layer,
#                                        input_dim=hp.dim_z, hidden_dim=hp.mixture_hidden_dim, output_dim=3,
#                                        learning_rate=hp.learning_rate_mixture)
#
# print("Misspecified Ising simulation takes %s seconds to finish." % (time.time() - start_time))
