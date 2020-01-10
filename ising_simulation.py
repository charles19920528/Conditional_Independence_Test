import numpy as np
import generate_train_fucntions as gt
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import multiprocessing as mp
from functools import partial
import pickle
import hyperparameters as hp


import simulation_functions as sf
import time

# Ising simulation
start_time = time.time()

sf.oracle_ising_simulation_loop(scenario = "null", result_dict_name = "ising")
sf.oracle_ising_simulation_loop(scenario = "alt", result_dict_name = "ising")

print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))

"""
def ising_simulation_wrapper(simulation_index, scenario, sample_size, epoch):
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt", dtype = np.float32)
    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}_{simulation_index}.txt", dtype = np.float32)

    ising_training_pool_instance = gt.IsingTrainingPool(z_mat=z_mat, hidden_1_out_dim=hp.hidden_1_out_dim,
                                                        learning_rate=hp.learning_rate, buffer_size=hp.buffer_size,
                                                        batch_size=hp.batch_size, epoch=epoch)

    predicted_parameter_mat = ising_training_pool_instance.trainning(x_y_mat = x_y_mat)

    print(f"{scenario}: Sample size {sample_size} simulation {simulation_index} is done.")

    return (simulation_index, predicted_parameter_mat)


###########################
# Simulate under the null #
###########################
start_time_null = time.time()

ising_result_null_dict = dict()

for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet):
    pool = mp.Pool(processes=hp.process_number)
    simulation_index_vet = range(hp.simulation_times)
    pool_result_vet = pool.map(partial(ising_simulation_wrapper, sample_size=sample_size, scenario="null"),
                               simulation_index_vet)

    ising_result_null_dict[sample_size] = dict(pool_result_vet)

with open("./results/ising_result_null_dict.p", "wb") as fp:
    pickle.dump(ising_result_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

end_time_null = time.time() - start_time_null

##################################
# Simulate under the alternative #
##################################
start_time_alt = time.time()

ising_result_alt_dict = dict()

for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet):
    pool = mp.Pool(processes=hp.process_number)
    simulation_index_vet = range(hp.simulation_times)
    pool_result_vet = pool.map(partial(ising_simulation_wrapper,
                                       sample_size=sample_size, scenario="alt"),
                               simulation_index_vet)

    ising_result_alt_dict[sample_size] = dict(pool_result_vet)

with open("./results/ising_result_alt_dict.p", "wb") as fp:
    pickle.dump(ising_result_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

end_time_alt = time.time() - start_time_alt

print(f"Null take {end_time_null} to train.")
print(f"Alt takes {end_time_alt} to train.")
"""


