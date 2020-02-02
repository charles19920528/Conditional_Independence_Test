import numpy as np
import generate_train_fucntions as gt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt

import pickle
import hyperparameters as hp

import multiprocessing as mp
from functools import partial


def tuning_pool_wrapper_ising(trail_index, scenario, sample_size, epoch, weights_dict, input_dim=3, hidden_1_out_dim=2,
                              hidden_2_out_dim=2):
    x_y_mat = np.loadtxt(f"./data/ising_data/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)
    z_mat = np.loadtxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)

    weights = weights_dict[sample_size][trail_index]
    true_network = gt.IsingNetwork(3, 3, 3)
    true_network.dummy_run()
    true_network.set_weights(weights)
    true_parameter_mat = true_network(z_mat)

    wrong_ising_network = gt.WrongIsingNetwork(input_dim=input_dim, hidden_1_out_dim=hidden_1_out_dim,
                                               hidden_2_out_dim=hidden_2_out_dim, output_dim=3)
    ising_tunning_instance = gt.IsingTunning(z_mat=z_mat, x_y_mat=x_y_mat,
                                             ising_network=wrong_ising_network, batch_size=100,
                                             max_epoch=int(epoch))

    result_mat = ising_tunning_instance.tuning(print_loss_boolean=True, true_parameter_mat=true_parameter_mat)

    return (trail_index, result_mat)


# trail_index = 3
# scenario = "alt"
# sample_size = 1000
# epoch = 20
#
# with open('data/ising_data/weights_dict.p', 'rb') as fp:
#     weights_dict = pickle.load(fp)
# tuning_pool_wrapper_ising(trail_index, scenario, sample_size, epoch, weights_dict)


def tuning_pool_wrapper_mixture(trail_index, scenario, hidden_1_out_dim, hidden_2_out_dim, sample_size, epoch):
    x_y_mat = np.loadtxt(f"./data/ising_data/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)
    z_mat = np.loadtxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)

    if scenario == "null":
        p_mat_true = np.repeat(0.25, sample_size * 4).reshape(sample_size, 4)
    else:
        p_mat_true = np.loadtxt(f"data/mixture_data/alt/p_mat/p_mat_alt_{sample_size}_{trail_index}.txt")

    mixture_ising_network = gt.MixutureIsingNetwork(input_dim=3, hidden_1_out_dim=hidden_1_out_dim,
                                                    hidden_2_out_dim=hidden_2_out_dim, output_dim=3)
    ising_tunning_instance = gt.IsingTunning(z_mat=z_mat, x_y_mat=x_y_mat,
                                             ising_network=mixture_ising_network, batch_size=100,
                                             max_epoch=int(epoch))

    result_mat = ising_tunning_instance.tuning(print_loss_boolean=True, p_mat_true=p_mat_true)

    return (trail_index, result_mat)


# tuning_pool_wrapper_mixture(trail_index, scenario, sample_size, epoch)

def tuning_loop(tunning_pool_wrapper, scenario, epoch_vet, trail_index_vet, result_dict_name, weights_dict=None,
                sample_size_vet=hp.sample_size_vet, process_number=4, hidden_1_out_dim_vet=np.repeat(2, 4),
                hidden_2_out_dim_vet=np.repeat(2, 4)):
    result_dict = dict()

    for sample_size, epoch, hidden_1_out_dim, hidden_2_out_dim in zip(sample_size_vet, epoch_vet, hidden_1_out_dim_vet,
                                                                      hidden_2_out_dim_vet):
        pool = mp.Pool(processes=process_number)
        if tunning_pool_wrapper == tuning_pool_wrapper_mixture:
            pool_result_vet = pool.map(partial(tunning_pool_wrapper, sample_size=sample_size, scenario=scenario,
                                               epoch=epoch, hidden_1_out_dim=hidden_1_out_dim,
                                               hidden_2_out_dim=hidden_2_out_dim), trail_index_vet)
        else:
            pool_result_vet = pool.map(partial(tunning_pool_wrapper, sample_size=sample_size, scenario=scenario,
                                               epoch=epoch, weights_dict=weights_dict), trail_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

        pool.close()
        pool.join()

    with open(f"tunning/{result_dict_name}_result_{scenario}_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


epoch_vet_ising =np.int32(hp.epoch_vet_misspecified * 1.3)
trail_index_vet = np.array([25, 26, 48, 210])
# Ising data_tuning
# with open('data/ising_data/weights_dict.p', 'rb') as fp:
#     weights_dict = pickle.load(fp)

# tuning_loop(tunning_pool_wrapper=tuning_pool_wrapper_ising, scenario="alt", epoch_vet=epoch_vet_ising,
#             trail_index_vet=np.arange(4), result_dict_name="ising_data", weights_dict=weights_dict)
# tuning_loop(tunning_pool_wrapper=tuning_pool_wrapper_ising, scenario="null", epoch_vet=epoch_vet_ising,
#             trail_index_vet=np.arange(4), result_dict_name="ising_data", weights_dict=weights_dict)

epoch_vet_mixture_alt = np.array([50, 20, 10, 10])
epoch_vet_mixture_null = np.array([50, 20, 10, 10])
tuning_loop(tunning_pool_wrapper=tuning_pool_wrapper_mixture, scenario="alt", epoch_vet=epoch_vet_mixture_alt,
            trail_index_vet=trail_index_vet, result_dict_name="mixture_data", hidden_1_out_dim_vet=
            np.array([3, 4, 6, 8]), hidden_2_out_dim_vet=np.array([3, 4, 6, 8]), process_number=4)

tuning_loop(tunning_pool_wrapper=tuning_pool_wrapper_mixture, scenario="null", epoch_vet=epoch_vet_mixture_null,
            trail_index_vet=trail_index_vet, result_dict_name="mixture_data", hidden_1_out_dim_vet=
            np.array([3, 4, 6, 8]), hidden_2_out_dim_vet=np.array([3, 4, 6, 8]), process_number=4)


###################
# Result analysis #
###################
def plot_loss_kl(experiment_result_dict, sample_size, end_epoch, start_epoch=0, plot_loss=True, plot_kl=True,
                 plot_test_loss=True):
    """
    Assuming there are only four sample size we are simulating. We plot the RoC curve and save the plot under
    ./results/plots/{model_for_main_title}.png.

    :param experiment_result_dict: A dictionary which is the output of the fpr_tpr function.
    :param model_for_main_title: A string ('str' class) which we use to name the graph to save as
    {model_for_main_title}.png.

    :return:
    None
    """
    trail_index_vet = experiment_result_dict[sample_size].keys()
    epoch_vet = np.arange(end_epoch)[start_epoch:]
    fig, ax = plt.subplots(4, 3)

    if not plot_kl and not plot_loss and not plot_test_loss:
        print("Nothing to print")
        return

    for i, trail_index in enumerate(trail_index_vet):
        for column in np.arange(3):
            ax[i, column].plot(epoch_vet, experiment_result_dict[sample_size][trail_index]["loss_array"][column,
                                          start_epoch:end_epoch])

    ax[0, 0].set_title("Training Likelihood")
    ax[0, 1].set_title("Test Likelihodd")
    ax[0, 2].set_title("KL")

    fig.suptitle(f"Sample size {sample_size}")
    fig.show()

#    fig.savefig(f"./tunning/plots/{sample_size}_loss_{plot_loss}_kl_{plot_kl}_test_{plot_test_loss}.png")


# with open(f"tunning/ising_data_result_alt_dict.p", "rb") as fp:
#     ising_data_result_alt_dict = pickle.load(fp)
#
# for sample_size, epoch in zip(hp.sample_size_vet, epoch_vet_ising):
#     plot_loss_kl(ising_data_result_alt_dict, sample_size, end_epoch=epoch)

# Mixture data
# Alt
with open(f"tunning/mixture_data_result_alt_dict.p", "rb") as fp:
    mixture_data_result_alt_dict = pickle.load(fp)

for sample_size, epoch in zip(hp.sample_size_vet, epoch_vet_mixture_alt):
    plot_loss_kl(mixture_data_result_alt_dict, sample_size, end_epoch=epoch)

# Null
with open(f"tunning/mixture_data_result_null_dict.p", "rb") as fp:
    mixture_data_result_null_dict = pickle.load(fp)

for sample_size, epoch in zip(hp.sample_size_vet, epoch_vet_mixture_null):
    plot_loss_kl(mixture_data_result_null_dict, sample_size, end_epoch=epoch)

