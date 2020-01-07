from CCIT import CCIT
import numpy as np
import hyperparameters as hp
import pickle

"""
sample_size = 30
scenario = "null"
simulation_index = 0
z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}.txt")
x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt")

#with 30 bootstrap iterations and 20 threads in parallel.
x = x_y_mat[:, 0][:, np.newaxis]
y = x_y_mat[:, 1][:, np.newaxis]
p_value = CCIT.CCIT(x, y, z_mat,num_iter = 30, bootstrap = True, nthread = 20)
"""

def process_x_y_mat(x_y_mat):
    x = x_y_mat[:, 0][:, np.newaxis]
    y = x_y_mat[:, 1][:, np.newaxis]
    return x, y

###########################
# Simulate under the null #
###########################
def ccit_simulation(scenario, sample_size_vet = hp.sample_size_vet, simulation_times = hp.simulation_times,
                    num_iter=30, bootstrap=True, nthread=20, **kwargs):
    for sample_size in sample_size_vet:
        ccit_result_dict = dict()
        z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}.txt")

        sample_size_result_dict = dict()
        for simulation_index in range(simulation_times):
            x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt")
            x_array, y_array = process_x_y_mat(x_y_mat)

            p_value = CCIT.CCIT(x_array, y_array, z_mat, num_iter=num_iter, bootstrap=bootstrap, nthread=nthread,
                                **kwargs)
            sample_size_result_dict[simulation_times] = p_value
            print(f"Scenario {scenario}, sample size {sample_size}, simulation {simulation_index} is finished")

        ccit_result_dict[sample_size] = sample_size_result_dict
        return ccit_result_dict


ccit_result_null_dict = ccit_simulation(scenario = "null")
ccit_result_alt_dict = ccit_simulation(scenario = "")

with open("./results/ccit_result_null_dict.p", "wb") as fp:
    pickle.dump(ccit_result_null_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)

with open("./results/ccit_result_alt_dict.p", "wb") as fp:
    pickle.dump(ccit_result_alt_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)