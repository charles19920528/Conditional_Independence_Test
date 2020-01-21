import time

import simulation_functions as sf
import generate_train_fucntions as gt
import hyperparameters as hp

# Naive Chisq simulation
sf.simulation_loop(simulation_wrapper = sf.naive_chisq_wrapper,  scenario = "null", result_dict_name = "naive_chisq")
sf.simulation_loop(simulation_wrapper = sf.naive_chisq_wrapper,  scenario = "alt", result_dict_name = "naive_chisq")


# Stratified Chisq simulation
sf.simulation_loop(simulation_wrapper = sf.stratified_chisq_wrapper, scenario = "null",
                   result_dict_name = "stratified_chisq")
sf.simulation_loop(simulation_wrapper = sf.stratified_chisq_wrapper, scenario = "alt",
                   result_dict_name = "stratified_chisq")

# Ising simulation
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "null",
                   result_dict_name = "ising", ising_network_class = gt.IsingNetwork,
                   input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim,
                   output_dim = 3)
sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "alt",
                   result_dict_name = "ising", ising_network_class = gt.IsingNetwork,
                   input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim,
                   output_dim = 3)

print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))


# CCIT simulation
process_number_ccit = 3
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "null", result_dict_name = "ccit",
                process_number = process_number_ccit)
sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "alt", result_dict_name = "ccit",
                process_number = process_number_ccit)

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))


# Misspecified Ising Model simulation
sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "null",
                   result_dict_name = "misspecified_ising", ising_network_class = gt.WrongIsingNetwork,
                   input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim_misspecified,
                   hidden_2_out_dim = hp.hidden_2_out_dim_misspecified, output_dim = 3)
sf.simulation_loop(simulation_wrapper = sf.ising_simulation_wrapper, scenario = "alt",
                   result_dict_name = "misspecified_ising", ising_network_class = gt.WrongIsingNetwork,
                   input_dim = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim_misspecified,
                   hidden_2_out_dim = hp.hidden_2_out_dim_misspecified, output_dim = 3)

