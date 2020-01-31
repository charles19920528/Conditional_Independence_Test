import time

import simulation_functions as sf
import generate_train_fucntions as gt
import hyperparameters as hp

##############
# Ising data #
##############
# Naive Chisq simulation
sf.simulation_loop(simulation_wrapper = sf.naive_chisq_wrapper,  scenario = "null", data_directory_name="ising_data",
                   result_dict_name = "naive_chisq", result_directory_name = "ising_data")

sf.simulation_loop(simulation_wrapper = sf.naive_chisq_wrapper,  scenario = "alt", data_directory_name="ising_data",
                   result_dict_name = "naive_chisq", result_directory_name = "ising_data")


# Stratified Chisq simulation
sf.simulation_loop(simulation_wrapper = sf.stratified_chisq_wrapper, scenario = "null",
data_directory_name="ising_data", result_dict_name = "stratified_chisq",
                   result_directory_name = "ising_data")

sf.simulation_loop(simulation_wrapper = sf.stratified_chisq_wrapper, scenario = "alt", data_directory_name="ising_data",
                   result_dict_name = "stratified_chisq", result_directory_name = "ising_data")

# Ising simulation
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "null",
                   data_directory_name="ising_data", result_dict_name = "ising", result_directory_name = "ising_data",
                   ising_network_class = gt.IsingNetwork, input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim,
                   output_dim = 3)

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "alt",
                   data_directory_name="ising_data", result_dict_name = "ising", result_directory_name = "ising_data",
                   ising_network_class = gt.IsingNetwork, input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim,
                   output_dim = 3)

print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))


# Ising simulation using residual statistic
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "null",
                   data_directory_name = "ising_data", result_dict_name = "ising_residual",
                   result_directory_name = "ising_data",
                   ising_network_class = gt.IsingNetwork, input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim,
                   output_dim = 2)

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "alt",
                   data_directory_name = "ising_data", result_dict_name = "ising_residual",
                   result_directory_name = "ising_data",
                   ising_network_class = gt.IsingNetwork, input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim,
                   output_dim = 2)

print("Ising (residuals) simulation takes %s seconds to finish." % (time.time() - start_time))

# CCIT simulation
process_number_ccit = 3
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "null", data_directory_name="ising_data",
                   result_dict_name = "ccit", result_directory_name = "ising_data",
                   process_number = process_number_ccit)

sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "alt", data_directory_name="ising_data",
                   result_dict_name = "ccit", result_directory_name = "ising_data",
                   process_number = process_number_ccit)

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))


# Misspecified Ising Model simulation
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "null",
                   data_directory_name="ising_data", result_dict_name = "misspecified_ising",
                   result_directory_name = "ising_data",
                   ising_network_class = gt.WrongIsingNetwork,
                   epoch_vet = hp.epoch_vet_misspecified, input_dim = hp.dim_z,
                   hidden_1_out_dim = hp.hidden_1_out_dim_misspecified,
                   hidden_2_out_dim = hp.hidden_2_out_dim_misspecified, output_dim = 3)

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "alt", data_directory_name="ising_data",
                   result_dict_name = "misspecified_ising", result_directory_name = "ising_data",
                   ising_network_class = gt.WrongIsingNetwork,
                   epoch_vet = hp.epoch_vet_misspecified, input_dim = hp.dim_z,
                   hidden_1_out_dim = hp.hidden_1_out_dim_misspecified,
                   hidden_2_out_dim = hp.hidden_2_out_dim_misspecified, output_dim = 3)

print("Misspecified Ising simulation takes %s seconds to finish." % (time.time() - start_time))

# Misspecified Ising simulation using residual statistic
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "null",
                   data_directory_name = "ising_data", result_dict_name = "misspecified_ising_residual",
                   result_directory_name = "ising_data",
                   ising_network_class = gt.WrongIsingNetwork,
                   epoch_vet = hp.epoch_vet_misspecified, input_dim = hp.dim_z,
                   hidden_1_out_dim = hp.hidden_1_out_dim_misspecified,
                   hidden_2_out_dim = hp.hidden_2_out_dim_misspecified, output_dim = 2)

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "alt",
                   data_directory_name = "ising_data", result_dict_name = "misspecified_ising_residual",
                   result_directory_name = "ising_data",
                   ising_network_class = gt.WrongIsingNetwork,
                   epoch_vet = hp.epoch_vet_misspecified, input_dim = hp.dim_z,
                   hidden_1_out_dim = hp.hidden_1_out_dim_misspecified,
                   hidden_2_out_dim = hp.hidden_2_out_dim_misspecified, output_dim = 2)

print("Ising (residuals) simulation takes %s seconds to finish." % (time.time() - start_time))


################
# Mixture data #
################
# Naive Chisq simulation
sf.simulation_loop(simulation_wrapper = sf.naive_chisq_wrapper,  scenario = "null", data_directory_name="mixture_data",
                   result_dict_name = "naive_chisq", result_directory_name = "mixture_data")

sf.simulation_loop(simulation_wrapper = sf.naive_chisq_wrapper,  scenario = "alt", data_directory_name="mixture_data",
                   result_dict_name = "naive_chisq", result_directory_name = "mixture_data")


# Stratified Chisq simulation
sf.simulation_loop(simulation_wrapper = sf.stratified_chisq_wrapper, scenario = "null",
                   data_directory_name="mixture_data", result_dict_name = "stratified_chisq",
                   result_directory_name = "mixture_data", cluster_number=2)

sf.simulation_loop(simulation_wrapper = sf.stratified_chisq_wrapper, scenario = "alt",
                   data_directory_name="mixture_data", result_dict_name = "stratified_chisq",
                   result_directory_name = "mixture_data", cluster_number=3)

# CCIT
process_number_ccit = 3
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "null", data_directory_name="mixture_data",
                   result_dict_name = "ccit", result_directory_name = "mixture_data",
                   process_number = process_number_ccit)

sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "alt", data_directory_name="mixture_data",
                   result_dict_name = "ccit", result_directory_name = "mixture_data",
                   process_number = process_number_ccit)

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))