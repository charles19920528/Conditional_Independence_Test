import os
import pickle
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import generate_train_functions_nightly as gt
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def ising_tuning_one_trial(trial_index, sample_size, scenario, data_directory_name, max_epoch, number_of_test_samples,
                         network_model_class, network_model_class_kwargs={}, learning_rate=hp.learning_rate,
                         true_weights_dict=None, cut_off_radius=None):
    """
    Fit the neural network belongs to the {network_model_class} on {trial_index}th trial of data with sample size
    {sample_size} and record the losses and kl-divergence.

    :param trial_index: An integer indicating the data is the {trial_index}th trial of sample size {sample_size}.
    :param sample_size: An integer indicating the sample size of the data.
    :param scenario: It should either be "null" or "alt" depending on if the data is generated under the null or
        alternative.
    :param data_directory_name: It should either be "ising_data" or "mixture_data" depending on if the data is generated
        under the Ising or mixture model.
    :param max_epoch: An integer indicating the number of times training process pass through the data set.
    :param number_of_test_samples: An integer which is the number of samples used as validation set.
    :param network_model_class: A subclass of tf.keras.Model with output dimension 3. This is the neural network to fit
        on the data.
    :param network_model_class_kwargs: A dictionary containing keyword arguments to create an instance of the
        network_model.
    :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
    :param cut_off_radius: If supplied, it should be a scalar which is the cut_off_radius used when generating
            the mixture data.
    :param true_weights_dict: If supplied, it should be the dictionary containing all true weights arrays of the data
        generating Ising network.

    :return:
    """
    assert cut_off_radius is None or true_weights_dict is None, \
        "Both cut_off_radius and weights_dic are supplied."
    assert cut_off_radius is not None or true_weights_dict is not None, \
        "Neither cut_off_radius nor tweights_dict are supplied."

    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt",
                         dtype=np.float32)
    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)

    network_instance = network_model_class(**network_model_class_kwargs)
    network_train_tune_instance = gt.NetworkTrainingTuning(z_mat=z_mat, x_y_mat=x_y_mat, network_model=network_instance,
                                                           max_epoch=max_epoch, learning_rate=learning_rate)

    assert scenario in ["null", "alt"], "scernaio has to be either null or alt."
    is_null_boolean = False
    if scenario == "null":
        is_null_boolean = True

    if true_weights_dict is not None:
        true_weights_array = true_weights_dict[sample_size][trial_index]
        loss_kl_array = network_train_tune_instance.tuning(print_loss_boolean=False, is_null_boolean=is_null_boolean,
                                                           number_of_test_samples=number_of_test_samples,
                                                           true_weights_array=true_weights_array)
    else:
        loss_kl_array = network_train_tune_instance.tuning(print_loss_boolean=False, is_null_boolean=is_null_boolean,
                                                           number_of_test_samples=number_of_test_samples,
                                                           cut_off_radius=cut_off_radius)

    print(f"Scenario: {scenario}, sample size: {sample_size}, trial: {trial_index} finished.")

    return (trial_index, loss_kl_array)


def tuning_wrapper(pool, scenario, data_directory_name, network_model_class, number_of_test_samples_vet, max_epoch_vet,
                trial_index_vet, result_dict_name, network_model_class_kwargs={}, sample_size_vet=hp.sample_size_vet,
                learning_rate=hp.learning_rate, weights_or_radius_kwargs={}):
    """
    A wrapper function uses multiprocess pool function to call the ising_tuning_one_trial functino on all data with
    sample size and trials specified by the arguments of the function.

    :param pool: A multiprocessing.pool.Pool instance.
    :param scenario: It should either be "null" or "alt" depending on if the data is generated under the null or
        alternative.
    :param data_directory_name: It should either be "ising_data" or "mixture_data" depending on if the data is generated
        under the Ising or mixture model.
    :param network_model_class: A subclass of tf.keras.Model with output dimension 3. This is the neural network to fit
        on the data.
    :param number_of_test_samples_vet: An array of integers which contains the sample size of data used.
    :param max_epoch_vet: An array of integers. It should have the same length as the sample_size_vet does.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.
    :param result_dict_name: A string ('str' class). The name of the result dictionary.
    :param network_model_class_kwargs: A dictionary containing keyword arguments to create an instance of the
        network_model.
    :param sample_size_vet: An array of integers which are the sample size of data used.
    :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
    :param weights_or_radius_kwargs: A dictionary whose key should either be "true_weights_dict" or "cut_off_radius".

    :return:
        result_dict: A dictionary. Keys are the sample size in the sample_size_vet. Each value is again a dictionary
        containing the loss_kl_array of each trial.
    """
    result_dict = dict()
    for sample_size, max_epoch, number_of_test_samples in zip(sample_size_vet, max_epoch_vet,
                                                              number_of_test_samples_vet):
        pool_result_vet = pool.map(partial(ising_tuning_one_trial, scenario=scenario, sample_size=sample_size,
                                           data_directory_name=data_directory_name, max_epoch=max_epoch,
                                           number_of_test_samples=number_of_test_samples,
                                           network_model_class=network_model_class,
                                           network_model_class_kwargs=network_model_class_kwargs,
                                           learning_rate=learning_rate, **weights_or_radius_kwargs), trial_index_vet)


        result_dict[sample_size] = dict(pool_result_vet)

    with open(f"tuning/raw_result_dict/{result_dict_name}_{scenario}_result_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


###################
# Result analysis #
###################
def optimal_epoch_kl_one_trial(trial_result_array):
    """
    Extra the epoch which gives the minimum kl-divergence on the test set along with minimum kl-divergence on one trial
    result.
    
    :param trial_result_array: An array which is the ouput of the tuning method of the NetworkTrainingTuning class.

    :return:
        An array of length 2. [argmin, min].
    """
    trial_kl_array = trial_result_array[2, :]

    return [np.argmin(trial_kl_array), np.min(trial_kl_array)]
#    return [np.argmin(trial_kl_array) + 1, np.min(trial_kl_array)]


def optimal_epoch_kl_one_sample_size(sample_size, trial_index_vet, tuning_result_dict):
    """
    Extra the epoch which gives the minimum kl-divergence on the test set along with minimum kl-divergence on trials
    given in the trial_index_vet. All trials have sample size {sample_size}.

    :param sample_size: An integer indicating the sample size of the trial data.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.
    :param tuning_result_dict: A dictionary which should be an output of the tuning_loop function.

    :return:
        1. sample_size. 2. epoch_kl_mat: A n by 3 matrix. n is the length of trial_index_vet. The columns of the matrix
        contains trial indices, the epoch which produces  the minimum kl and the minimum kl in such order.

    """
    number_of_trials = len(trial_index_vet)
    epoch_kl_mat = np.zeros((number_of_trials, 3))

    for i, trial_index in enumerate(trial_index_vet):
        epoch_kl_mat[i, 0] = trial_index
        epoch_kl_mat[i, 1:3] = optimal_epoch_kl_one_trial(tuning_result_dict[sample_size][trial_index])

    return sample_size, epoch_kl_mat


def optimal_epoch_kl(pool, sample_size_vet, trial_index_vet, tuning_result_dict, raw_result_dict_name):
    """
    Extract the epoch which gives the minimum kl-divergence on the test set along with minimum kl-divergence for all
    trials and sample size. The function will also print the mean and sd of the kl-divergence.

    :param pool:
    :param sample_size_vet:
    :param trial_index_vet:
    :param tuning_result_dict: A dictionary which should be an output of the tuning_loop function.

    :return:
    """
    pool_result_vet = pool.map(partial(optimal_epoch_kl_one_sample_size, tuning_result_dict=tuning_result_dict,
                                       trial_index_vet=trial_index_vet), sample_size_vet)

    optimal_epoch_kl_dict = dict(pool_result_vet)

    kl_vet = [optimal_epoch_kl_dict[sample_size][:, 2] for sample_size in sample_size_vet]

    mean_kl_vet = np.array([np.mean(kl_array) for kl_array in kl_vet])
    sd_kl_vet = np.array([np.std(kl_array) for kl_array in kl_vet])

    print(f"Settings: {raw_result_dict_name}")
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
    fig.savefig(f"./tuning/epoch_kl_graph/{figure_name}.png")


def process_plot_epoch_kl_raw_dict(pool, raw_result_dict_name, sample_size_vet, trial_index_vet):
    """

    :param pool:
    :param scenario:
    :param name_dictionary: raw_result_dict_name, number_forward_layers, hidden_dim, scenario,
    :param sample_size_vet:
    :param trial_index_vet:
    :return:
    """
    with open(f"tuning/raw_result_dict/{raw_result_dict_name}_result_dict.p", "rb") as fp:
        tuning_result_dict = pickle.load(fp)

    optimal_epoch_kl_mat_dict = optimal_epoch_kl(pool=pool, sample_size_vet=sample_size_vet,
                                                 trial_index_vet=trial_index_vet, tuning_result_dict=tuning_result_dict,
                                                 raw_result_dict_name=raw_result_dict_name)

    plot_optimal_epoch_kl(optimal_epoch_kl_mat_dict, figure_name=raw_result_dict_name)

    with open(f"./tuning/optimal_epoch/{raw_result_dict_name}_epoch_kl_mat_dict.p", "wb") as fp:
        pickle.dump(optimal_epoch_kl_mat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def plot_loss_kl(scenario, raw_result_dict_name, trial_index_vet, sample_size, end_epoch, start_epoch=0,
                 plot_train_loss=True, plot_kl=True, plot_test_loss=True):
    """
    Plot training loss, test loss, and kl-divergence on four trials with the same sample_size.

    :param scenario: It should either be "null" or "alt" depending on if the data is generated under the null or
        alternative.
    :param raw_result_dict_name: A string ('str' class). The first part of the name of the raw result dictionary.
    :param trial_index_vet: An array of four integers. Losses /kl on these trials will be plotted.
    :param sample_size: An integer indicating the sample size of the data.
    :param end_epoch: An integer. The right end of x_axis of the plot.
    :param start_epoch:
    :param plot_train_loss: A
    :param plot_kl:
    :param plot_test_loss:
    :return:
    """
    with open(f"tuning/raw_result_dict/{raw_result_dict_name}_{scenario}_result_dict.p", "rb") as fp:
        experiment_result_dict = pickle.load(fp)

    epoch_vet = np.arange(end_epoch)[start_epoch:]
    fig, ax = plt.subplots(4, 3)

    if not plot_kl and not plot_train_loss and not plot_test_loss:
        print("Nothing to print")
        return

    for i, trial_index in enumerate(trial_index_vet):
        for column in np.arange(3):
            ax[i, column].plot(epoch_vet, experiment_result_dict[sample_size][trial_index][column,
                                          start_epoch:end_epoch])

    ax[0, 0].set_title("Training Likelihood")
    ax[0, 1].set_title("Test Likelihodd")
    ax[0, 2].set_title("KL")

    fig.suptitle(f"{scenario}, sample size {sample_size}")
    fig.show()
