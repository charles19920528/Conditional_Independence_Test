import numpy as np
import generate_train_fucntions as gt

import os
import multiprocessing as mp
from functools import partial
import pickle
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def ising_simulation_wrapper(simulation_index, ising_training_pool_instance, scenario, sample_size):
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt")
    predicted_parameter_mat = ising_training_pool_instance.trainning(x_y_mat = x_y_mat)

    print(f"Sample size {sample_size} simulation {simulation_index} is done.")

    return (simulation_index, predicted_parameter_mat)


###########################
# Simulate under the null #
###########################
ising_result_null_dict = dict()

for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet):
    z_mat = np.loadtxt("./data/z_mat/z_mat_%d.txt" % sample_size, dtype="float32")
    ising_training_pool_instance = gt.IsingTrainingPool(z_mat=z_mat, hidden_1_out_dim=hp.hidden_1_out_dim,
                                            learning_rate=hp.learning_rate, buffer_size=hp.buffer_size,
                                                    batch_size=hp.batch_size, epoch=epoch)

    pool = mp.Pool(processes=hp.process_number)
    simulation_index_vet = range(hp.simulation_times)
    pool_result_vet = pool.map(partial(ising_simulation_wrapper,
                                       ising_training_pool_instance=ising_training_pool_instance,
                                       sample_size=sample_size, scenario="null"),
                               simulation_index_vet)

    ising_result_null_dict[sample_size] = dict(pool_result_vet)

with open("./results/ising_result_null_dict.p", "wb") as fp:
    pickle.dump(ising_result_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


##################################
# Simulate under the alternative #
##################################
ising_result_alt_dict = dict()

for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet):
    z_mat = np.loadtxt("./data/z_mat/z_mat_%d.txt" % sample_size, dtype="float32")
    ising_training_pool_instance = gt.IsingTrainingPool(z_mat=z_mat, hidden_1_out_dim=hp.hidden_1_out_dim,
                                            learning_rate=hp.learning_rate, buffer_size=hp.buffer_size,
                                                    batch_size=hp.batch_size, epoch=epoch)

    pool = mp.Pool(processes=hp.process_number)
    simulation_index_vet = range(hp.simulation_times)
    pool_result_vet = pool.map(partial(ising_simulation_wrapper,
                                       ising_training_pool_instance=ising_training_pool_instance,
                                       sample_size=sample_size, scenario="alt"),
                               simulation_index_vet)

    ising_result_alt_dict[sample_size] = dict(pool_result_vet)

with open("./results/ising_result_alt_dict.p", "wb") as fp:
    pickle.dump(ising_result_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

