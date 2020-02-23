import os
import pickle
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import generate_train_fucntions as gt
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def tuning_pool_wrapper_ising_data(trail_index, scenario, sample_size, epoch, number_of_test_samples,
                                   weights_dict, number_forward_elu_layers, input_dim, hidden_dim, output_dim):
    x_y_mat = np.loadtxt(f"./data/ising_data/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)
    z_mat = np.loadtxt(f"./data/ising_data/z_mat/z_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)

    true_weights_array = weights_dict[sample_size][trail_index]

    wrong_ising_network = gt.FullyConnectedNetwork(number_forward_elu_layers=number_forward_elu_layers,
                                                   input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    ising_tunning_instance = gt.IsingTrainingTunning(z_mat=z_mat, x_y_mat=x_y_mat, ising_network=wrong_ising_network,
                                                     max_epoch=epoch)

    assert scenario in ["null", "alt"], "scernaio has to be either null or alt."
    is_null_boolean = False
    if scenario == "null":
        is_null_boolean = True

    result_mat = ising_tunning_instance.tuning(print_loss_boolean=False, is_null_boolean=is_null_boolean,
                                               number_of_test_samples=number_of_test_samples,
                                               true_weight_array=true_weights_array)

    return (trail_index, result_mat)


def tuning_pool_wrapper_mixture_data(trail_index, scenario, sample_size, epoch,
                                     number_forward_elu_layers, input_dim, hidden_dim, output_dim):

    x_y_mat = np.loadtxt(f"./data/mixture_data/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)
    z_mat = np.loadtxt(f"./data/mixture_data/z_mat/z_mat_{sample_size}_{trail_index}.txt", dtype=np.float32)

    if scenario == "null":
        p_mat_true = np.repeat(0.25, sample_size * 4).reshape(sample_size, 4)
    else:
        p_mat_true = np.loadtxt(f"data/mixture_data/alt/p_mat/p_mat_alt_{sample_size}_{trail_index}.txt")

    ising_network_instance = gt.FullyConnectedNetwork(number_forward_elu_layers=number_forward_elu_layers,
                                                      input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    ising_tunning_instance = gt.IsingTunning(z_mat=z_mat, x_y_mat=x_y_mat, ising_network=ising_network_instance,
                                             batch_size=hp.batch_size, max_epoch=int(epoch))

    result_mat = ising_tunning_instance.tuning(print_loss_boolean=True, p_mat_true=p_mat_true)

    return (trail_index, result_mat)


def tuning_loop(pool, tunning_pool_wrapper, scenario, number_of_test_samples_vet, number_forward_elu_layers, input_dim,
                hidden_dim, output_dim, epoch_vet, trail_index_vet, result_dict_name,
                sample_size_vet=hp.sample_size_vet, **kwargs):

    result_dict = dict()
    for sample_size, epoch, number_of_test_samples in zip(sample_size_vet, epoch_vet, number_of_test_samples_vet):
#        pool = mp.Pool(processes=process_number)
        pool_result_vet = pool.map(partial(tunning_pool_wrapper, scenario=scenario, sample_size=sample_size,
                                           epoch=epoch, number_of_test_samples=number_of_test_samples,
                                           number_forward_elu_layers=number_forward_elu_layers,
                                           input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs),
                                   trail_index_vet)

            # pool_result_vet = pool.map(partial(tunning_pool_wrapper, sample_size=sample_size, scenario=scenario,
            #                                    epoch=epoch, number_forward_elu_layers=number_forward_elu_layers,
            #                                        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim),
            #                            trail_index_vet)
            #
            # pool_result_vet = pool.map(partial(tunning_pool_wrapper, sample_size=sample_size, scenario=scenario,
            #                                    epoch=epoch, weights_dict=weights_dict, number_forward_elu_layers=number_forward_elu_layers,
            #                                        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim), trail_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

        # pool.close()
        # pool.join()

    with open(f"tunning/{result_dict_name}_result_{scenario}_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


###################
# Result analysis #
###################
def optimal_epoch_kl_one_trail(trail_result_dict):
    trail_kl_array = trail_result_dict["loss_array"][2, :]

    return [np.argmin(trail_kl_array) + 1, np.min(trail_kl_array)]


def optimal_epoch_kl_one_sample_size(sample_size, trail_index_vet, tuning_result_dict):
    number_of_trails = len(trail_index_vet)
    epoch_kl_mat = np.zeros((number_of_trails, 3))

    for i, trail_index in enumerate(trail_index_vet):
        epoch_kl_mat[i, 0] = trail_index
        epoch_kl_mat[i, 1:3] = optimal_epoch_kl_one_trail(tuning_result_dict[sample_size][trail_index])

    return sample_size, epoch_kl_mat


def optimal_epoch_kl(sample_size_vet, trail_index_vet, tuning_result_dict, process_number=4):
    pool = mp.Pool(processes=process_number)
    pool_result_vet = pool.map(partial(optimal_epoch_kl_one_sample_size, tuning_result_dict=tuning_result_dict,
                                       trail_index_vet=trail_index_vet), sample_size_vet)

    optimal_epoch_kl_dict = dict(pool_result_vet)
    return optimal_epoch_kl_dict

def plot_optimal_epoch_kl(epoch_kl_dict):
    sample_size_vet = list(epoch_kl_dict.keys())
#    number_of_trails = epoch_kl_dict[sample_size_vet[0]].shape[0]
#    color_vet = ["teal", "coral", "slategray", "blueviolet"]
    fig, ax = plt.subplots(1, 2)

    kl_vet = [epoch_kl_dict[sample_size][:, 2] for sample_size in sample_size_vet]
    epoch_vet = [epoch_kl_dict[sample_size][:, 1] for sample_size in sample_size_vet]
    ax[0].boxplot(epoch_vet, labels=sample_size_vet)
    ax[0].set_title("Epoch")

    ax[1].boxplot(kl_vet, labels=sample_size_vet)
    ax[1].set_title("KL")

    mean_kl_vet = np.array([np.mean(kl_array) for kl_array in kl_vet])
    sd_kl_vet = np.array([np.std(kl_array) for kl_array in kl_vet])

    print(f"Mean kls are {mean_kl_vet}")
    print(f"Std of kls are {sd_kl_vet}")


def process_plot_epoch_kl_raw_dict(path_epoch_kl_dict, sample_size_vet, trail_index_vet):
    with open(f"{path_epoch_kl_dict}", "rb") as fp:
        epoch_kl_raw_dict = pickle.load(fp)

    epoch_kl_mat_dict = optimal_epoch_kl(sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet,
                                         tuning_result_dict=epoch_kl_raw_dict)

    plot_optimal_epoch_kl(epoch_kl_mat_dict)

    return epoch_kl_mat_dict


def plot_loss_kl(experiment_result_dict, trail_index_vet, sample_size, end_epoch, start_epoch=0, plot_loss=True,
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
