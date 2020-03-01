import os
import pickle
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import generate_train_fucntions as gt
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def ising_tuning_wrapper(trail_index, scenario, data_directory_name, sample_size, epoch, number_of_test_samples,
                         ising_network, learning_rate=hp.learning_rate, weights_dict=None, cut_off_radius=None,**kwargs):
    """

    :param trail_index:
    :param scenario:
    :param sample_size:
    :param epoch:
    :param number_of_test_samples:
    :param ising_network:
    :param learning_rate:
    :param weights_dict:
    :param cut_off_radius:
    :param kwargs:
    :return:
    """
    assert cut_off_radius is None or weights_dict is None, \
        "Both cut_off_radius and weights_dic are supplied."
    assert cut_off_radius is not None or weights_dict is not None, \
        "Neither cut_off_radius nor tweights_dict are supplied."

    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt",
                         dtype=np.float32)
    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)

    ising_network_instance = ising_network(**kwargs)
    ising_tunning_instance = gt.IsingTrainingTunning(z_mat=z_mat, x_y_mat=x_y_mat, ising_network=ising_network_instance,
                                                     max_epoch=epoch, learning_rate=learning_rate)

    assert scenario in ["null", "alt"], "scernaio has to be either null or alt."
    is_null_boolean = False
    if scenario == "null":
        is_null_boolean = True

    if weights_dict is not None:
        true_weights_array = weights_dict[sample_size][trail_index]
        result_dict = ising_tunning_instance.tuning(print_loss_boolean=False, is_null_boolean=is_null_boolean,
                                                    number_of_test_samples=number_of_test_samples,
                                                    true_weight_array=true_weights_array)
    else:
        result_dict = ising_tunning_instance.tuning(print_loss_boolean=False, is_null_boolean=is_null_boolean,
                                                   number_of_test_samples=number_of_test_samples,
                                                   cut_off_radius=cut_off_radius)

    print(f"Scenario: {scenario}, sample size: {sample_size}, trail: {trail_index} finished.")

    return (trail_index, result_dict)


def tuning_loop(pool, scenario, data_directory_name, number_of_test_samples_vet, ising_network,
                epoch_vet, trail_index_vet, result_dict_name, sample_size_vet=hp.sample_size_vet, **kwargs):
    """

    :param pool:
    :param scenario:
    :param number_of_test_samples_vet:
    :param ising_network:
    :param epoch_vet:
    :param trail_index_vet:
    :param result_dict_name:
    :param sample_size_vet:
    :param kwargs:
    :return:
    """

    result_dict = dict()
    for sample_size, epoch, number_of_test_samples in zip(sample_size_vet, epoch_vet, number_of_test_samples_vet):

        pool_result_vet = pool.map(partial(ising_tuning_wrapper, scenario=scenario, sample_size=sample_size,
                                           data_directory_name=data_directory_name,
                                           epoch=epoch, number_of_test_samples=number_of_test_samples,
                                           ising_network=ising_network, **kwargs), trail_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

    with open(f"tunning/{result_dict_name}_result_{scenario}_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


###################
# Result analysis #
###################
def optimal_epoch_kl_one_trail(trail_result_dict):
    """

    :param trail_result_dict:
    :return:
    """
    trail_kl_array = trail_result_dict["loss_array"][2, :]

    return [np.argmin(trail_kl_array) + 1, np.min(trail_kl_array)]


def optimal_epoch_kl_one_sample_size(sample_size, trail_index_vet, experiment_result_dict):
    """

    :param sample_size:
    :param trail_index_vet:
    :param experiment_result_dict:
    :return:
    """
    number_of_trails = len(trail_index_vet)
    epoch_kl_mat = np.zeros((number_of_trails, 3))

    for i, trail_index in enumerate(trail_index_vet):
        epoch_kl_mat[i, 0] = trail_index
        epoch_kl_mat[i, 1:3] = optimal_epoch_kl_one_trail(experiment_result_dict[sample_size][trail_index])

    return sample_size, epoch_kl_mat


def optimal_epoch_kl(pool, sample_size_vet, trail_index_vet, experiment_result_dict):
    """

    :param pool:
    :param sample_size_vet:
    :param trail_index_vet:
    :param experiment_result_dict:
    :return:
    """
    pool_result_vet = pool.map(partial(optimal_epoch_kl_one_sample_size, experiment_result_dict=experiment_result_dict,
                                       trail_index_vet=trail_index_vet), sample_size_vet)

    optimal_epoch_kl_dict = dict(pool_result_vet)

    kl_vet = [optimal_epoch_kl_dict[sample_size][:, 2] for sample_size in sample_size_vet]

    mean_kl_vet = np.array([np.mean(kl_array) for kl_array in kl_vet])
    sd_kl_vet = np.array([np.std(kl_array) for kl_array in kl_vet])

    print(f"Mean kls are {mean_kl_vet}")
    print(f"Std of kls are {sd_kl_vet}")

    return optimal_epoch_kl_dict


def plot_optimal_epoch_kl(optimal_epoch_kl_dict, figure_name):
    """

    :param optimal_epoch_kl_dict:
    :param figure_name:
    :return:
    """
    sample_size_vet = list(optimal_epoch_kl_dict.keys())
    fig, ax = plt.subplots(1, 2)

    kl_vet = [optimal_epoch_kl_dict[sample_size][:, 2] for sample_size in sample_size_vet]
    epoch_vet = [optimal_epoch_kl_dict[sample_size][:, 1] for sample_size in sample_size_vet]

    ax[0].boxplot(epoch_vet, labels=sample_size_vet)
    ax[0].set_title("Epoch")

    ax[1].boxplot(kl_vet, labels=sample_size_vet)
    ax[1].set_title("KL")

    fig.suptitle(figure_name)
    fig.savefig(f"./tunning/epoch_kl_graph/{figure_name}.png")




def process_plot_epoch_kl_raw_dict(pool, scenario, result_dict_name, sample_size_vet, trail_index_vet):
    """

    :param pool:
    :param scenario:
    :param result_dict_name:
    :param sample_size_vet:
    :param trail_index_vet:
    :return:
    """
    with open(f"tunning/{result_dict_name}_result_{scenario}_dict.p", "rb") as fp:
        experiment_result_dict = pickle.load(fp)

    optimal_epoch_kl_mat_dict = optimal_epoch_kl(pool=pool, sample_size_vet=sample_size_vet,
                                                 trail_index_vet=trail_index_vet,
                                                 experiment_result_dict=experiment_result_dict)

    plot_optimal_epoch_kl(optimal_epoch_kl_mat_dict, figure_name=f"{result_dict_name} {scenario}")

    with open(f"./tunning/optimal_epoch/{result_dict_name}_{scenario}_epoch_kl_mat_dict.p", "wb") as fp:
        pickle.dump(optimal_epoch_kl_mat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # return optimal_epoch_kl_mat_dict


def plot_loss_kl(scenario, result_dict_name, trail_index_vet, sample_size, end_epoch, start_epoch=0, plot_loss=True,
                 plot_kl=True, plot_test_loss=True):
    """

    :param experiment_result_dict:
    :param trail_index_to_plot_vet:
    :param sample_size:
    :param end_epoch:
    :param start_epoch:
    :param plot_loss:
    :param plot_kl:
    :param plot_test_loss:
    :return:
    """
    with open(f"tunning/{result_dict_name}_result_{scenario}_dict.p", "rb") as fp:
        experiment_result_dict = pickle.load(fp)

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

    fig.suptitle(f"{scenario}, sample size {sample_size}")
    fig.show()
