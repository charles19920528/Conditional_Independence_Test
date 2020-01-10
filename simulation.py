import simulation_functions as sf
import time

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

sf.oracle_ising_simulation_loop(scenario = "null", result_dict_name = "ising")
sf.oracle_ising_simulation_loop(scenario = "alt", result_dict_name = "ising")

print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))


# CCIT simulation
process_number_ccit = 3
start_time = time.time()

sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "null", result_dict_name = "ccit",
                process_number = process_number_ccit)
sf.simulation_loop(simulation_wrapper = sf.ccit_wrapper, scenario = "alt", result_dict_name = "ccit",
                process_number = process_number_ccit)

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))

