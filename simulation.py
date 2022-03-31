import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import os

import simulation_functions as sf
import generate_train_functions as gt
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

pool = mp.Pool(processes=hp.process_number)

##########################
# Naive Chi squared test #
##########################
# Ising data
sf.simulation_loop(pool=pool, simulation_method=sf.naive_chisq_method, scenario="null",
                   data_directory_name="ising_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=pool, simulation_method=sf.naive_chisq_method, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

# Mixture data
sf.simulation_loop(pool=pool, simulation_method=sf.naive_chisq_method, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=pool, simulation_method=sf.naive_chisq_method, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="naive_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

###############################
# Stratified Chi squared test #
###############################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

# Ising data
sf.simulation_loop(pool=pool, simulation_method=sf.stratified_chisq_method, scenario="null",
                   data_directory_name="ising_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=pool, simulation_method=sf.stratified_chisq_method, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials))

# Mixture data
sf.simulation_loop(pool=pool, simulation_method=sf.stratified_chisq_method, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials), cluster_number=2)

sf.simulation_loop(pool=pool, simulation_method=sf.stratified_chisq_method, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="stratified_chisq",
                   trial_index_vet=np.arange(hp.number_of_trials), cluster_number=2)

###############
# Ising Model #
###############
scenario_vet = ["alt"]

# Mixture data
mixture_number_forward_layer_vet = [hp.mixture_number_forward_layer_null, hp.mixture_number_forward_layer_alt]
mixture_hidden_dim_vet = [hp.mixture_hidden_dim_null, hp.mixture_hidden_dim_alt]
mixture_epoch_vet_vet = [hp.full_model_mixture_epoch_vet_null, hp.full_model_mixture_epoch_vet_alt]

for test_sample_prop in hp.test_prop_list:
    for mixture_number_forward_layer, mixture_hidden_dim, mixture_epoch_vet, scenario in \
            zip(mixture_number_forward_layer_vet, mixture_hidden_dim_vet, mixture_epoch_vet_vet, scenario_vet):
        mixture_result_dict_name = f"mixture_data_{mixture_number_forward_layer}_{mixture_hidden_dim}_breg"
        mixture_network_model_class_kwargs = {"number_forward_layers": mixture_number_forward_layer,
                                              "input_dim": hp.dim_z, "hidden_dim": mixture_hidden_dim, "output_dim": 3}
        mixture_network_model_class_kwargs_vet = [mixture_network_model_class_kwargs for _ in
                                                  range(len(hp.sample_size_vet))]

        np.random.seed(hp.seed_index)
        tf.random.set_seed(hp.seed_index)

        start_time = time.time()
        sf.ising_simulation_loop(pool=pool, scenario=scenario, data_directory_name="mixture_data",
                                 result_dict_name=mixture_result_dict_name,
                                 trial_index_vet=np.arange(hp.number_of_trials),
                                 network_model_class=gt.FullyConnectedNetwork,
                                 network_model_class_kwargs_vet=mixture_network_model_class_kwargs_vet,
                                 epoch_vet=mixture_epoch_vet, learning_rate=hp.learning_rate_mixture,
                                 sample_size_vet=hp.sample_size_vet,
                                 test_sample_prop=test_sample_prop)

        print(f"Mixture data, test prop: {test_sample_prop}, scenario: {scenario}")
        print(f"Ising simulation on mixture data takes %s seconds to finish." % (time.time() - start_time))




# Ising data
true_result_dict_name = f"ising_data_true_architecture_breg"
true_network_model_class_kwargs = {"number_forward_layers": 1, "input_dim": hp.dim_z,
                                   "hidden_dim": hp.hidden_1_out_dim, "output_dim": 3}
true_network_model_class_kwargs_vet = [true_network_model_class_kwargs for _ in range(len(hp.sample_size_vet))]

for scenario in scenario_vet:
    np.random.seed(hp.seed_index)
    tf.random.set_seed(hp.seed_index)

    start_time = time.time()
    sf.ising_simulation_loop(pool=pool, scenario=scenario, data_directory_name="ising_data",
                             result_dict_name=true_result_dict_name,
                             trial_index_vet=np.arange(hp.number_of_trials),
                             network_model_class=gt.FullyConnectedNetwork,
                             network_model_class_kwargs_vet=true_network_model_class_kwargs_vet,
                             epoch_vet=hp.ising_epoch_vet, learning_rate=hp.learning_rate,
                             test_sample_prop=0.1)

    print(f"Ising Data, test prop: 0.1, scenario: {scenario}")
    print(f"Ising simulation on ising data takes %s seconds to finish." % (time.time() - start_time))
#
# for test_sample_prop in hp.test_prop_list:
#     # Alternative
#     np.random.seed(hp.seed_index)
#     tf.random.set_seed(hp.seed_index)
#
#     start_time = time.time()
#
#     sf.ising_simulation_loop(pool=pool, scenario="alt", data_directory_name="ising_data",
#                              result_dict_name=true_result_dict_name, trial_index_vet=np.arange(hp.number_of_trials),
#                              network_model_class=gt.FullyConnectedNetwork,
#                              network_model_class_kwargs_vet=true_network_model_class_kwargs_vet,
#                              epoch_vet=hp.ising_epoch_vet, learning_rate=hp.learning_rate_mixture,
#                              sample_size_vet=hp.sample_size_vet, test_sample_prop=test_sample_prop)
#
#     print("Ising simulation under null Ising data takes %s seconds to finish." % (time.time() - start_time))
#
#     # Null
#     np.random.seed(hp.seed_index)
#     tf.random.set_seed(hp.seed_index)
#
#     start_time = time.time()
#     sf.ising_simulation_loop(pool=pool, scenario="null", data_directory_name="ising_data",
#                              result_dict_name=true_result_dict_name, trial_index_vet=np.arange(hp.number_of_trials),
#                              network_model_class=gt.FullyConnectedNetwork,
#                              network_model_class_kwargs_vet=true_network_model_class_kwargs_vet,
#                              epoch_vet=hp.ising_epoch_vet, learning_rate=hp.learning_rate_mixture,
#                              sample_size_vet=hp.sample_size_vet, test_samples_prop=test_sample_prop)
#
#     print("Ising simulation under alternative Ising data takes %s seconds to finish." % (time.time() - start_time))


pool.close()
pool.join()

########
# CCIT #
########
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

process_number_ccit = 3
ccit_pool = mp.Pool(processes=process_number_ccit)

# Ising data
start_time = time.time()

sf.simulation_loop(pool=ccit_pool, simulation_method=sf.ccit_method, scenario="null",
                   data_directory_name="ising_data", result_dict_name="ccit",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=ccit_pool, simulation_method=sf.ccit_method, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="ccit",
                   trial_index_vet=np.arange(hp.number_of_trials))

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))

# mixture data
start_time = time.time()

sf.simulation_loop(pool=ccit_pool, simulation_method=sf.ccit_method, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="ccit",
                   trial_index_vet=np.arange(hp.number_of_trials))

sf.simulation_loop(pool=ccit_pool, simulation_method=sf.ccit_method, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="ccit",
                   trial_index_vet=np.arange(hp.number_of_trials))

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))

ccit_pool.close()
ccit_pool.join()