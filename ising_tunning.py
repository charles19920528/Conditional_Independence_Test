import numpy as np
import generate_train_fucntions as gt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt

import pickle
import hyperparameters as hp

import multiprocessing as mp
from functools import partial

trail_index = 3
scenario = "alt"
sample_size = 30
epoch = 1
with open('data/ising_data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)

def tuning_pool_wrapper_ising(trail_index, scenario, sample_size, epoch, weights_dict):
    x_y_mat = np.loadtxt(f"./data/ising_data/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)
    z_mat = np.loadtxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)

    weights = weights_dict[sample_size][trail_index]
    true_network = gt.IsingNetwork(3, 3, 3)
    true_network.dummy_run()
    true_network.set_weights(weights)
    true_parameter_mat = true_network(z_mat)

    wrong_ising_network = gt.WrongIsingNetwork(input_dim=3, hidden_1_out_dim=2, hidden_2_out_dim=2,
                                               output_dim=3)
    ising_tunning_instance = gt.IsingTunning(z_mat=z_mat, x_y_mat=x_y_mat,
                                             ising_network=wrong_ising_network, batch_size=100,
                                             max_epoch=int(epoch))

    result_mat = ising_tunning_instance.tuning(print_loss_boolean=True, true_parameter_mat=true_parameter_mat)

    return (trail_index, result_mat)

tuning_pool_wrapper_ising(trail_index, scenario, sample_size, epoch, weights_dict)


def tuning_pool_wrapper_mixture(trail_index, scenario, sample_size, epoch):
    x_y_mat = np.loadtxt(f"./data/ising_data/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)
    z_mat = np.loadtxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)

    if scenario == "null":
        p_mat_true = np.repeat(0.25, sample_size * 4).reshape(sample_size, 4)
    else:
        p_mat_true = np.loadtxt(f"data/mixture_data/alt/p_mat/p_mat_alt_{sample_size}_{trail_index}.txt")

    wrong_ising_network = gt.WrongIsingNetwork(input_dim=3, hidden_1_out_dim=2, hidden_2_out_dim=2,
                                               output_dim=3)
    ising_tunning_instance = gt.IsingTunning(z_mat=z_mat, x_y_mat=x_y_mat,
                                             ising_network=wrong_ising_network, batch_size=100,
                                             max_epoch=int(epoch))

    result_mat = ising_tunning_instance.tuning(print_loss_boolean=True, p_mat_true=p_mat_true)

    return (trail_index, result_mat)

tuning_pool_wrapper_mixture(trail_index, scenario, sample_size, epoch)

def tuning_loop(tunning_pool_wrapper, scenario, epoch_vet, trail_index_vet, result_dict_name,
                sample_size_vet=hp.sample_size_vet, process_number=hp.process_number):
    result_dict = dict()
    for sample_size, epoch in zip(sample_size_vet, epoch_vet):
        pool = mp.Pool(processes=process_number)

        pool_result_vet = pool.map(partial(tunning_pool_wrapper, sample_size=sample_size, scenario=scenario,
                                           epoch=epoch),
                                   trail_index_vet)

        result_dict[sample_size] = pool_result_vet

    with open(f"tunning/{result_dict_name}_result_{scenario}_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)






# Ising data_tuning
with open('data/ising_data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)

"""
scenario = "alt"
trail_index_vet = np.arange(4)
"""


"""
experiment_result_dict = dict()
for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet_misspecified * 1.3):
    pool = mp.Pool(processes=hp.process_number)
    trail_index_vet = range(hp.number_of_trails)

    pool_result_vet = pool.map(partial(simulation_wrapper, sample_size=sample_size, scenario=scenario,
                                       data_directory_name=data_directory_name, epoch=epoch, **kwargs),
                               trail_index_vet)




    experiment_result_dict[sample_size] = sample_size_result_dict
"""


























"""

experiment_result_dict = dict()
for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet_misspecified):
    sample_size_result_dict = dict()
    for trail in trail_index_vet:
        x_y_mat = np.loadtxt(f"./data/ising_data/{scenario}/x_y_mat_{sample_size}_{trail}.txt", dtype=np.float32)
        z_mat = np.loadtxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{trail}.txt", dtype=np.float32)

        weights = weights_dict[sample_size][trail]
        true_network = gt.IsingNetwork(3, 3, 3)
        true_network.dummy_run()
        true_network.set_weights(weights)
        true_parameter_mat = true_network(z_mat)

        wrong_ising_network = gt.WrongIsingNetwork(input_dim=3, hidden_1_out_dim=2, hidden_2_out_dim=2, output_dim=3)
        ising_tunning_instance = gt.IsingTunning(z_mat=z_mat, x_y_mat=x_y_mat,
                                                 ising_network=wrong_ising_network, batch_size = 100,
                                                 max_epoch = int(epoch * 1.2))

        sample_size_result_dict[trail] = ising_tunning_instance.tuning(print_loss_boolean=True,
                                                                       true_parameter_mat=true_parameter_mat)

    experiment_result_dict[sample_size] = sample_size_result_dict

with open(f"./tunning/experiment_result_dict.p", "wb") as fp:
    pickle.dump(experiment_result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
"""


###################
# Result analysis #
###################
def plot_loss_kl(experiment_result_dict, sample_size, start_epoch = 0,plot_loss = True, plot_kl = True,
                 plot_test_loss = True):
    """
    Assuming there are only four sample size we are simulating. We plot the RoC curve and save the plot under
    ./results/plots/{model_for_main_title}.png.

    :param experiment_result_dict: A dictionary which is the output of the fpr_tpr function.
    :param model_for_main_title: A string ('str' class) which we use to name the graph to save as
    {model_for_main_title}.png.

    :return:
    None
    """
    epoch_vet = np.arange(experiment_result_dict[sample_size][0]["loss_array"][0, :].shape[0])[start_epoch:]
    fig, ax = plt.subplots(4, 3)

    if not plot_kl and not plot_loss and not plot_test_loss:
        print("Nothing to print")
        return


    ax[0, 0].plot(epoch_vet, experiment_result_dict[sample_size][0]["loss_array"][0, start_epoch:])
    ax[0, 0].set_title("Training Loss")
    ax[1, 0].plot(epoch_vet, experiment_result_dict[sample_size][1]["loss_array"][0, start_epoch:])
    ax[2, 0].plot(epoch_vet, experiment_result_dict[sample_size][2]["loss_array"][0, start_epoch:])
    ax[3, 0].plot(epoch_vet, experiment_result_dict[sample_size][3]["loss_array"][0, start_epoch:])


    ax[0, 1].plot(epoch_vet, experiment_result_dict[sample_size][0]["loss_array"][1, start_epoch:])
    ax[0, 1].set_title("Training KL")
    ax[1, 1].plot(epoch_vet, experiment_result_dict[sample_size][1]["loss_array"][1, start_epoch:])
    ax[2, 1].plot(epoch_vet, experiment_result_dict[sample_size][2]["loss_array"][1, start_epoch:])
    ax[3, 1].plot(epoch_vet, experiment_result_dict[sample_size][3]["loss_array"][1, start_epoch:])


    ax[0, 2].plot(epoch_vet, experiment_result_dict[sample_size][0]["loss_array"][2, start_epoch:])
    ax[0, 2].set_title("Test Loss")
    ax[1, 2].plot(epoch_vet, experiment_result_dict[sample_size][1]["loss_array"][2, start_epoch:])
    ax[2, 2].plot(epoch_vet, experiment_result_dict[sample_size][2]["loss_array"][2, start_epoch:])
    ax[3, 2].plot(epoch_vet, experiment_result_dict[sample_size][3]["loss_array"][2, start_epoch:])


    fig.suptitle(f"Sample size {sample_size}")
    fig.show()
#    fig.savefig(f"./tunning/plots/{sample_size}_loss_{plot_loss}_kl_{plot_kl}_test_{plot_test_loss}.png")


with open(f"./tunning/experiment_result_dict.p", "rb") as fp:
    experiment_result_dict = pickle.load(fp)
"""
for sample_size in hp.sample_size_vet:
    plot_loss_kl(experiment_result_dict, sample_size, plot_loss=False, start_epoch=0)

"""

