from functools import partial
import pickle
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from CCIT import CCIT

import generate_train_functions_nightly as gt
import hyperparameters as hp



####################
# Simulation loops #
####################
def simulation_loop(pool, simulation_wrapper, scenario, data_directory_name, result_dict_name,
                    sample_size_vet=hp.sample_size_vet):
    """
    A wrap up function for the simulation loop using the multiprocessing Pool function. The function will
    save the result dictionary in a pickle file under ./results/{result_dict_name}_result_{scenario}_dict.p.

    :param pool: A multiprocessing.pool.Pool instance.
    :param simulation_wrapper: A function which should one of the wrapper functions defined below except for the
        ising_simulation_wrapper.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
    under the null or alternative hypothesis.
    :param data_directory_name: A string ('str' class) of the path towards the simulation data.
    :param result_dict_name:  A string ('str' class) which we use to name the result dictionary as
    {result_dict_name}_result_{scenario}_dict.
    :param sample_size_vet: A python list of integers. It contains all the sample size we simulated.
    :param number_of_trials: An integer which is the number of trials we simulate for each sample size
    :param epoch_vet: A python list of integers. It provides the training epoch, if simulation wrapper is the
    ising_simulation_wrapper,
    :param process_number: An integer. It is the argument of the Pool function telling us the number of workers.
    :param **kwargs: Keyword arguments to be passed in to the ising_simulation_wrapper when simulation_wrapper is
        ising_simulation_wrapper. See the documentation of ising_simulation_wrapper for additional details.

    :return:
        None.
    """
    result_dict = dict()

    for sample_size in sample_size_vet:
        trial_index_vet = range(number_of_trials)
        if simulation_wrapper == ising_simulation_wrapper and ising_simulation_wrapper_kwargs == {}:
            sys.exit("ising_simulation_wrapper_kwargs is not supplied while running the simulation of Ising Model.")
        else:
            pool_result_vet = pool.map(partial(simulation_wrapper, sample_size=sample_size, scenario=scenario,
                                               data_directory_name=data_directory_name,
                                               **ising_simulation_wrapper_kwargs), trial_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

        print(f"{result_dict_name}, {scenario}, {sample_size} finished")

    with open(f"./results/result_dict/{data_directory_name}/{result_dict_name}_{scenario}_result_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def ising_simulation_loop(pool, simulation_wrapper, scenario, data_directory_name, result_dict_name,
                    sample_size_vet=hp.sample_size_vet, number_of_trials=hp.number_of_trials,
                    epoch_vet=hp.epoch_vet, ising_simulation_wrapper_kwargs={}):




# def simulation_loop_ising_optimal_epoch(pool, epoch_kl_dict_name, scenario, data_directory_name,
#                                         ising_network_class, sample_size_vet=hp.sample_size_vet,
#                                         number_of_trials=hp.number_of_trials,
#                                         number_of_test_samples_vet=hp.number_of_test_samples_vet, **kwargs):
#
#     with open(f"tuning/optimal_epoch/{epoch_kl_dict_name}_{scenario}_epoch_kl_mat_dict.p", "rb") as fp:
#         epoch_kl_dict = pickle.load(fp)
#
#     result_dict = dict()
#     for sample_size, number_of_test_samples in zip(sample_size_vet, number_of_test_samples_vet):
#         trial_index_vet = range(number_of_trials)
#         epoch_vet = epoch_kl_dict[sample_size][:, 1].astype(np.int8)
#
#         pool_result_vet = pool.starmap(partial(ising_simulation_wrapper, sample_size=sample_size, scenario=scenario,
#                                                data_directory_name=data_directory_name,
#                                                ising_network_class=ising_network_class,
#                                                number_of_test_samples=number_of_test_samples, **kwargs),
#                                        zip(trial_index_vet, epoch_vet))
#
#         result_dict[sample_size] = dict(pool_result_vet)
#
#         print(f"{data_directory_name}, {scenario}, {sample_size} finished")
#
#     with open(f"./results/result_dict/{data_directory_name}/{epoch_kl_dict_name}_result_{scenario}_dict.p", "wb") as fp:
#         pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)




#####################
# Wrapper functions #
#####################
# Ising simulation
def ising_simulation_wrapper(trial_index, sample_size, scenario, data_directory_name, max_epoch, network_model_class,
                             number_of_test_samples, learning_rate=hp.learning_rate, network_model_class_kwargs={}):
    """
    A wrapper function for the multiprocessing Pool function. It will be passed into the simulation_loop function.
    This function uses the fit Neural Ising model to compute the test statistic and record the indices of test samples
    on {trial_index}th trial with sample size {sample_size} under the {scenario} hypothesis.
    The tuple returned return will be used to create a dictionary.

    :param trial_index: An integer indicating the data is the {trial_index}th trial of sample size {sample_size}.
    :param sample_size: An integer indicating the sample size of the data.
    :param scenario: It should either be "null" or "alt" depending on if the data is generated under the null or
        alternative.
    param data_directory_name: It should either be "ising_data" or "mixture_data" depending on if the data is generated
        under the Ising or mixture model.
    :param max_epoch: An integer indicating the number of training epoch when training the neural network.
    :param network_model_class: A subclass of tf.keras.Model with output dimension 3. An instance of the class is the
        neural network to fit on the data.
    :param number_of_test_samples: An integer which is the number of samples used as the test data.
    :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
    :param network_model_class_kwargs: Keyword arguments to be passed in to the constructor of the network_model_class.
    :return:

        A tuple (trial_index, result_vet). result_vet is the return of the gt.IsingTrainingPool function.
    """
    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt",
                         dtype=np.float32)
    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)

    network_model = network_model_class(**network_model_class_kwargs)
    training_tuning_instance = gt.NetworkTrainingTuning(z_mat=z_mat, x_y_mat=x_y_mat, network_model=network_model,
                                                        learning_rate=learning_rate, max_epoch=max_epoch)

    result_dict = training_tuning_instance.train_compute_test_statistic(print_loss_boolean=False,
                                                                         number_of_test_samples=number_of_test_samples)

    print(f"{scenario}: Sample size {sample_size} simulation {trial_index} is done.")

    return (trial_index, result_dict)


# Naive Chisq
def chi_squared_test(x_y_mat):
    """
    Perform unconditional Chi-squared test.

    :param x_y_mat: A 2-d numpy array with columns corresponding to x and y.

    :return:
    result_vet: A length 2 tuple. result_vet[0] is the test statistic and result_vet[1] is the p-value.
    """
    x_vet = x_y_mat[:, 0]
    y_vet = x_y_mat[:, 1]
    contingency_table = pd.crosstab(x_vet, y_vet, rownames="x", colnames="y")

    result_vet = chi2_contingency(contingency_table)[0:2]
    return result_vet


def naive_chisq_wrapper(trial_index, scenario, data_directory_name, sample_size):
    """
    A wrapper function for the multiprocessing Pool function. It will be passed into the partial function.
    The pool function will run iterations in parallel given a sample size and a scenario.
    This function perform Chi squared test on {trial_index}th trial with sample size
    {sample_size} under the {scenario} hypothesis.
    The return will be used to create a dictionary.

    :param trial_index: An integer indicating {trial_index}th trial among simulations under sample size {sample_size}.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
    under the null or alternative hypothesis.
    :param data_directory_name: A string ('str' class) of the path towards the simulation data.
    :param sample_size: An integer.

    :return:
    A tuple (trial_index, result_vet). result_vet is the return of the chi_squared_test function.
    """
    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt")
    result_vet = chi_squared_test(x_y_mat)
    return (trial_index, result_vet)


# Stratified Chisq
def stratify_x_y_mat(x_y_mat, z_mat, cluster_number=2):
    """
    Cluster data into {cluster_number} of clusters.

    :param x_y_mat: An n x 2 numpy array. Each row is the response of the ith observation.
    First column corresponds to x.
    :param z_mat: A 2D numpy array. Each row is a data point.
    :param cluster_number: An integer which is the number of clusters to form by the KMeans function.

    :return:
    z_mat_vet: A python list of length {cluster_number}. Each element is a 2D numpy array of a data cluster.
    """
    kmeans_model = KMeans(n_clusters=cluster_number, random_state=0)
    kmeans_result_vet = kmeans_model.fit(z_mat).labels_
    x_y_mat_vet = [x_y_mat[kmeans_result_vet == i, :] for i in range(cluster_number)]

    return x_y_mat_vet


def stratified_chi_squared_test(x_y_mat_vet):
    """
    Compute sum of the Chi squared statistic on each strata.

    :param x_y_mat_vet: A python list which is the output of the stratify_x_y_mat function.

    :return:
    A numeric which is the sum of of the chi-squared statistic on each stratum.
    """
    chi_square_statistic_vet = np.zeros(len(x_y_mat_vet))
    for iteration, x_y_mat in enumerate(x_y_mat_vet):
        chi_square_statistic_vet[iteration] = chi_squared_test(x_y_mat)[0]

    return sum(chi_square_statistic_vet)


def stratified_chisq_wrapper(trial_index, scenario, data_directory_name, sample_size, cluster_number=2):
    """
    A wrapper function for the multiprocessing Pool function. It will be passed into the partial function.
    The pool function will run iteration in parallel given a sample size and a scenario.
    This function perform stratified Chi squared test on {trial_index}th trial with sample size
    {sample_size} under the {scenario} hypothesis.
    The return will be used to create a dictionary.

    :param trial_index: An integer indicating {trial_index}th trial among simulations under sample size {sample_size}.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
    under the null or alternative hypothesis.
    :param sample_size: An integer.
    :param data_directory_name: A string ('str' class) of the path towards the simulation data.

    :return:
    A tuple (trial_index, result_vet). result_vet is the return of the stratified_chi_squared_test function.
    """
    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt")
    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)

    x_y_mat_vet = stratify_x_y_mat(x_y_mat=x_y_mat, z_mat=z_mat, cluster_number=cluster_number)
    test_statistic = stratified_chi_squared_test(x_y_mat_vet)

    return (trial_index, test_statistic)


# CCIT
def process_x_y_mat(x_y_mat):
    """
    Process the data so that the input can be fed in to the CCIT.CCIT function.

    :param x_y_mat: An n x 2 numpy array. Each row is the response of the ith observation.
    First column corresponds to x.

    :return:
    x: An n x 1 numpy array corresponding to X.
    y: An n x 1 numpy array corresponding to Y.
    """
    x = x_y_mat[:, 0][:, np.newaxis]
    y = x_y_mat[:, 1][:, np.newaxis]
    return x, y


def ccit_wrapper(trial_index, scenario, data_directory_name, sample_size, **kwargs):
    """
    A wrapper function for the multiprocessing Pool function. It will be passed into the partial function.
    The pool function will run iteration in parallel given a sample size and a scenario.
    This function perform the model powered conditional independence test proposed by  on {trial_index}th trial with
    sample size {sample_size} under the {scenario} hypothesis.
    The return will be used to create a dictionary.

    :param trial_index: An integer indicating {trial_index}th trial among simulations under sample size {sample_size}.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
    under the null or alternative hypothesis.
    :param data_directory_name: A string ('str' class) of the path towards the simulation data.
    :param sample_size: An integer.
    :param kwargs: Arguments for the CCIT.CCIT functions.

    :return:
    A tuple (trial_index, result_vet). result_vet is the return of the CCIT.CCIT function.
    """
    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt")
    x_array, y_array = process_x_y_mat(x_y_mat)
    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)

    p_value = CCIT.CCIT(x_array, y_array, z_mat, **kwargs)

    print(f"{scenario}: Sample size {sample_size} simulation {trial_index} is done.")

    return (trial_index, p_value)

