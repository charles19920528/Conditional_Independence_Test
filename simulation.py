import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import os

import simulation_functions as sf
import generate_train_fucntions as gt
import hyperparameters as hp

pool = mp.Pool(processes=hp.process_number)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


##########################
# Naive Chi squared test #
##########################
# Ising data
sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="null",
                   data_directory_name="ising_data", result_dict_name="naive_chisq")

sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="naive_chisq")

# Naive Chisq simulation
sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="naive_chisq")

sf.simulation_loop(pool=pool, simulation_wrapper=sf.naive_chisq_wrapper, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="naive_chisq")

###############################
# Stratified Chi squared test #
###############################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

# Ising data
sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="null",
                   data_directory_name="ising_data", result_dict_name="stratified_chisq")

sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="alt",
                   data_directory_name="ising_data", result_dict_name="stratified_chisq")

# Mixture data
sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="null",
                   data_directory_name="mixture_data", result_dict_name="stratified_chisq", cluster_number=2)

sf.simulation_loop(pool=pool, simulation_wrapper=sf.stratified_chisq_wrapper, scenario="alt",
                   data_directory_name="mixture_data", result_dict_name="stratified_chisq", cluster_number=2)


####################
# True Ising model #
####################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

start_time = time.time()

sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_true_rate_{hp.learning_rate}",
                                       scenario="null", data_directory_name="ising_data",
                                       ising_network_class=gt.IsingNetwork, result_dict_name="ising_true",
                                       input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_true_rate_{hp.learning_rate}",
                                       scenario="alt", data_directory_name="ising_data",
                                       ising_network_class=gt.IsingNetwork, result_dict_name="ising_true",
                                       input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))


#######################
# General Ising Model #
#######################
# Ising data
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

start_time = time.time()

sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_wrong_rate_{hp.learning_rate}",
                                       scenario="null", data_directory_name="ising_data",
                                       ising_network_class=gt.FullyConnectedNetwork, result_dict_name="ising_wrong",
                                       number_forward_elu_layers=hp.wrong_number_forward_elu_layer,
                                       input_dim=hp.dim_z, hidden_dim=hp.wrong_hidden_dim, output_dim=3)


sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name=f"ising_wrong_rate_{hp.learning_rate}",
                                       scenario="alt", data_directory_name="ising_data",
                                       ising_network_class=gt.FullyConnectedNetwork, result_dict_name="ising_wrong",
                                       number_forward_elu_layers=hp.wrong_number_forward_elu_layer,
                                       input_dim=hp.dim_z, hidden_dim=hp.wrong_hidden_dim, output_dim=3)

print("Misspecified Ising simulation takes %s seconds to finish." % (time.time() - start_time))

# Mixture data
epoch_kl_dict_name =f"mixture_{hp.mixture_number_forward_elu_layer}_{hp.mixture_hidden_dim}"

np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

start_time = time.time()

sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name="ising_wrong", scenario="null",
                                       data_directory_name="ising_data", ising_network_class=gt.FullyConnectedNetwork,
                                       result_dict_name=epoch_kl_dict_name,
                                       number_forward_elu_layers=hp.mixture_number_forward_elu_layer,
                                       input_dim=hp.dim_z, hidden_dim=hp.mixture_hidden_dim, output_dim=3)

sf.simulation_loop_ising_optimal_epoch(pool=pool, epoch_kl_dict_name="ising_wrong", scenario="alt",
                                       data_directory_name="ising_data", ising_network_class=gt.FullyConnectedNetwork,
                                       result_dict_name=epoch_kl_dict_name,
                                       number_forward_elu_layers=hp.mixture_number_forward_elu_layer,
                                       input_dim=hp.dim_z, hidden_dim=hp.mixture_hidden_dim, output_dim=3)

print("Misspecified Ising simulation takes %s seconds to finish." % (time.time() - start_time))

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
