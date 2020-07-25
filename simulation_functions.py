from functools import partial
import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from CCIT import CCIT

import generate_train_functions as gt
import hyperparameters as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


####################
# Simulation loops #
####################
def simulation_loop(pool, simulation_method, scenario, data_directory_name, result_dict_name, trial_index_vet,
                    sample_size_vet=hp.sample_size_vet, **kwargs):
    """
    A wrapper function uses the multiprocessing Pool function to use simulation_method on multiple data in parallel. The
    function will save the result dictionary in a pickle file under the path
    ./results/ with file name {result_dict_name}_result_{scenario}_dict.p.

    :param pool: A multiprocessing.pool.Pool instance.
    :param simulation_method: A function which should one of the simulation method functions defined below except for
        the ising_simulation_method function.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
        under the null or alternative hypothesis.
    :param data_directory_name: A string ('str' class) of the path towards the simulation data.
    :param result_dict_name:  A string ('str' class) which we use to name the result dictionary as
        {data_directory_name}/{result_dict_name}_{scenario}_result_dict.p.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.
    :param sample_size_vet: A python list of integers. It contains all the sample size we simulated.
    :param kwargs: Additional keywords arguments to pass into the simulation_method if necessary.

    :return:
        None.
    """
    result_dict = dict()

    for sample_size in sample_size_vet:
        pool_result_vet = pool.map(partial(simulation_method, sample_size=sample_size, scenario=scenario,
                                           data_directory_name=data_directory_name, **kwargs), trial_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

        print(f"{result_dict_name}, {scenario}, {sample_size} finished")

    with open(f"./results/result_dict/{data_directory_name}/{result_dict_name}_{scenario}_result_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def ising_simulation_loop(pool, scenario, data_directory_name, result_dict_name, trial_index_vet, network_model_class,
                          network_model_class_kwargs_vet, epoch_vet, learning_rate, sample_size_vet=hp.sample_size_vet,
                          number_of_test_samples_vet=hp.number_of_test_samples_vet):
    """
    A wrapper function uses the multiprocessing Pool function to use tbe ising_simulation_method on multiple data in
    parallel. The function will save the result dictionary in a pickle file under the path
    ./results/{data_directory_name}/ with file name {result_dict_name}_{scenario}_result_dict.p.

    :param pool: A multiprocessing.pool.Pool instance.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
        under the null or alternative hypothesis.
    :param data_directory_name: A string ('str' class) of the path towards the simulation data.
    :param result_dict_name:  A string ('str' class) which we use to name the result dictionary as
        {data_directory_name}/{result_dict_name}_{scenario}_result_dict.p.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.
    :param network_model_class: A subclass of tf.keras.Model with output dimension 3. An instance of the class is the
        neural network to fit on the data.
    :param network_model_class_kwargs_vet: An array of dictionaries. The array should have the same length as the
        sample_size_vet does. Each dictionary contains keyword arguments to create an instance of the
        network_model_class.
    :param epoch_vet: An array of integers. The array should have the same length as the sample_size_vet does.
        Each entry specifies the number of epochs we train the network on data with the corresponding sample size.
    :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
    :param sample_size_vet: A python list of integers. It contains all the sample size we simulated.
    :param number_of_test_samples_vet: An array of integers which contains the sample size of data used.

    :return:
        None
    """
    result_dict = dict()

    for sample_size, number_of_test_samples, epoch, network_model_class_kwargs in zip(sample_size_vet,
                                                                                      number_of_test_samples_vet,
                                                                                      epoch_vet,
                                                                                      network_model_class_kwargs_vet):
        pool_result_vet = pool.map(partial(ising_simulation_method, sample_size=sample_size, scenario=scenario,
                                           data_directory_name=data_directory_name, epoch=epoch,
                                           network_model_class=network_model_class,
                                           number_of_test_samples=number_of_test_samples, learning_rate=learning_rate,
                                           network_model_class_kwargs=network_model_class_kwargs), trial_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

        print(f"{result_dict_name}, {scenario}, sample size: {sample_size} finished")

    with open(f"./results/result_dict/{data_directory_name}/{result_dict_name}_{scenario}_result_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def ising_bootstrap_loop(pool, scenario, data_directory_name, ising_simulation_result_dict_name, result_dict_name,
                         trial_index_vet, network_model_class,
                         network_model_class_kwargs_vet, number_of_bootstrap_samples, max_epoch_vet,
                         learning_rate=hp.learning_rate, buffer_size = hp.buffer_size, batch_size = hp.batch_size,
                         sample_size_vet=hp.sample_size_vet):
    result_dict = {}
    for sample_size, network_model_class_kwargs, max_epoch in zip(sample_size_vet, network_model_class_kwargs_vet,
                                                                  max_epoch_vet):
        sample_size_result_dict = {}
        for trial_index in trial_index_vet:
            trial_result_dict = \
                ising_bootstrap_method(pool=pool, trial_index=trial_index, sample_size=sample_size, scenario=scenario,
                                       data_directory_name=data_directory_name,
                                       ising_simulation_result_dict_name=ising_simulation_result_dict_name,
                                       network_model_class=network_model_class,
                                       network_model_class_kwargs=network_model_class_kwargs,
                                       number_of_bootstrap_samples=number_of_bootstrap_samples, max_epoch=max_epoch,
                                       batch_size=batch_size, buffer_size=buffer_size, learning_rate=learning_rate)
            sample_size_result_dict[trial_index] = trial_result_dict
            print(f"Bootstrap, data: {data_directory_name}, scenario: {scenario}, sample_size: {sample_size}, "
                  f"trial_index: {trial_index} finished.")
            print(f"P-value is {trial_result_dict['p_value']}")

        result_dict[sample_size] = sample_size_result_dict

    with open(f"./results/result_dict/{data_directory_name}/{result_dict_name}_{scenario}_result_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


# Not in use.
# def argmax_simulation_loop(pool, trial_index_vet, sample_size_vet, scenario, data_directory_name,
#                            ising_simulation_result_dict_name, result_dict_name, network_model_class,
#                            network_model_class_kwargs_vet, network_net_size, number_of_nets):
#     """
#
#     :param result_dict_name:
#     :param pool:
#     :param trial_index_vet:
#     :param sample_size_vet:
#     :param scenario:
#     :param data_directory_name:
#     :param ising_simulation_result_dict_name:
#     :param network_model_class:
#     :param network_model_class_kwargs_vet:
#     :param network_net_size:
#     :param number_of_nets:
#     :return:
#     """
#     result_dict = {}
#     for sample_size, network_model_class_kwargs in zip(sample_size_vet, network_model_class_kwargs_vet):
#         sample_size_result_dict = {}
#         for trial_index in trial_index_vet:
#             sample_size_result_dict[trial_index] = \
#                 argmax_gaussian_process_simulation_method(pool=pool, trial_index=trial_index, sample_size=sample_size,
#                                                           scenario=scenario, data_directory_name=data_directory_name,
#                                                           ising_simulation_result_dict_name=ising_simulation_result_dict_name,
#                                                           network_model_class=network_model_class,
#                                                           network_model_class_kwargs=network_model_class_kwargs,
#                                                           network_net_size=network_net_size,
#                                                           number_of_nets=number_of_nets)
#         result_dict[sample_size] = sample_size_result_dict
#
#     with open(f"./results/result_dict/{data_directory_name}/{result_dict_name}_{scenario}_result_dict.p", "wb") as fp:
#         pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     return result_dict


#####################
# Wrapper functions #
#####################
# Ising simulation
def ising_simulation_method(trial_index, sample_size, scenario, data_directory_name, epoch, network_model_class,
                            network_model_class_kwargs, number_of_test_samples, learning_rate=hp.learning_rate):
    """
    This function fit a Neural Ising model to compute the test statistic and record the indices of test samples
    on {trial_index}th trial with sample size {sample_size} under the {scenario} hypothesis.
    The tuple returned return will be used to create a dictionary.

    :param data_directory_name: It should either be "ising_data" or "mixture_data" depending on if the data is generated
        under the Ising or mixture model.
    :param trial_index: An integer indicating the data is the {trial_index}th trial of sample size {sample_size}.
    :param sample_size: An integer indicating the sample size of the data.
    :param scenario: It should be either "null" or "alt" depending on if the data is generated under the null or
        alternative.
    param data_directory_name: It should either be "ising_data" or "mixture_data" depending on if the data is generated
        under the Ising or mixture model.
    :param epoch: An integer indicating the number of training epoch when training the neural network.
    :param network_model_class: A subclass of tf.keras.Model with output dimension 3. An instance of the class is the
        neural network to fit on the data.
    :param network_model_class_kwargs: Keyword arguments to be passed in to the constructor of the network_model_class.
    :param number_of_test_samples: An integer which is the number of samples used as the test data.
    :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.

    :return:
        A tuple (trial_index, result_vet). The result_vet is the output of the class method train_compute_test_statistic
        of the gt.NetworkTrainingTuning class.
    """
    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt",
                         dtype=np.float32)
    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)

    training_tuning_instance = gt.NetworkTrainingTuning(z_mat=z_mat, x_y_mat=x_y_mat,
                                                        network_model_class=network_model_class,
                                                        network_model_class_kwargs=network_model_class_kwargs,
                                                        learning_rate=learning_rate, epoch=epoch)

    result_dict = training_tuning_instance.train_compute_test_statistic(print_loss_boolean=False,
                                                                        number_of_test_samples=number_of_test_samples)

    print(f"Scenario: {scenario} Sample size: {sample_size} trial: {trial_index} is done.")

    return (trial_index, result_dict)


# Ising bootstrap
def ising_bootstrap_one_trial(_, fitted_train_p_mat, z_mat, train_indices_vet, test_indices_vet,
                              network_model_class, network_model_class_kwargs, buffer_size, batch_size, learning_rate,
                              max_epoch):
    """

    :param _:
    :param fitted_train_p_mat:
    :param z_mat:
    :param train_indices_vet:
    :param test_indices_vet:
    :param network_model_class:
    :param network_model_class_kwargs:
    :param buffer_size:
    :param batch_size:
    :param learning_rate:
    :param max_epoch:
    :return:
    """
    # Resample
    new_train_x_y_mat = gt.generate_x_y_mat(fitted_train_p_mat)
    train_ds = tf.data.Dataset.from_tensor_slices((z_mat[train_indices_vet, :], new_train_x_y_mat))
    train_ds = train_ds.shuffle(buffer_size).batch(batch_size)

    # Train the network.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    network_model = network_model_class(**network_model_class_kwargs)
    epoch = 0
    while epoch < max_epoch:
        _ = gt.train_network(train_ds=train_ds, optimizer=optimizer, network_model=network_model)
        epoch += 1

    predicted_test_parameter_mat = network_model(z_mat[test_indices_vet, :])
    jxy_squared_vet = np.square(predicted_test_parameter_mat[:, 2])
    jxy_squared_mean = np.mean(jxy_squared_vet)

    return jxy_squared_mean


def ising_bootstrap_method(pool, trial_index, sample_size, scenario, data_directory_name,
                           ising_simulation_result_dict_name, network_model_class, network_model_class_kwargs,
                           number_of_bootstrap_samples, max_epoch, batch_size=hp.batch_size, buffer_size=hp.buffer_size,
                           learning_rate=hp.learning_rate):
    """

    :param pool:
    :param trial_index:
    :param sample_size:
    :param scenario:
    :param data_directory_name:
    :param ising_simulation_result_dict_name:
    :param network_model_class:
    :param network_model_class_kwargs:
    :param number_of_bootstrap_samples:
    :param max_epoch:
    :param batch_size:
    :param buffer_size:
    :param learning_rate:
    :return:
    """

    with open(f'results/result_dict/{data_directory_name}/{ising_simulation_result_dict_name}_{scenario}_result_'
              f'dict.p', 'rb') as fp:
        ising_simulation_loop_result_dict = pickle.load(fp)

    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)
    fitted_train_p_mat = ising_simulation_loop_result_dict[sample_size][trial_index]["fitted_train_p_mat"]
    train_indices_vet = ising_simulation_loop_result_dict[sample_size][trial_index]["train_indices_vet"]
    test_indices_vet = ising_simulation_loop_result_dict[sample_size][trial_index]["test_indices_vet"]
    test_statistic = ising_simulation_loop_result_dict[sample_size][trial_index]["test_statistic"]

    bootstrap_test_statistic_vet = pool.map(partial(ising_bootstrap_one_trial, fitted_train_p_mat=fitted_train_p_mat,
                                                    z_mat=z_mat, train_indices_vet=train_indices_vet,
                                                    test_indices_vet=test_indices_vet,
                                                    network_model_class=network_model_class,
                                                    network_model_class_kwargs=network_model_class_kwargs,
                                                    buffer_size=buffer_size, batch_size=batch_size,
                                                    learning_rate=learning_rate, max_epoch=max_epoch),
                                            np.arange(number_of_bootstrap_samples))

    p_value = sum(np.array(bootstrap_test_statistic_vet) > test_statistic) / number_of_bootstrap_samples
    result_dict = {"p_value": p_value, "bootstrap_test_statistic_vet": bootstrap_test_statistic_vet,
                   "test_statistic": test_statistic}

    return result_dict


# not in use.
# def argmax_gaussian_process_simulation_method(pool, trial_index, sample_size, scenario, data_directory_name,
#                                               ising_simulation_result_dict_name, network_model_class,
#                                               network_model_class_kwargs, network_net_size, number_of_nets):
#     """
#
#     :param pool:
#     :param trial_index:
#     :param sample_size:
#     :param scenario:
#     :param data_directory_name:
#     :param ising_simulation_result_dict_name:
#     :param network_model_class:
#     :param network_model_class_kwargs:
#     :param network_net_size:
#     :param number_of_nets:
#     :return:
#     """
#     with open(f'results/result_dict/{data_directory_name}/{ising_simulation_result_dict_name}_{scenario}_result_'
#               f'dict.p', 'rb') as fp:
#         ising_simulation_loop_result_dict = pickle.load(fp)
#
#     trial_test_statistic = ising_simulation_loop_result_dict[sample_size][trial_index]["test_statistic"]
#     test_indices_vet = ising_simulation_loop_result_dict[sample_size][trial_index]["test_indices_vet"]
#
#     z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)
#     test_z_mat = z_mat[test_indices_vet, :]
#
#     test_statistic_sample_vet = \
#         gt.argmax_gaussian_process_one_trial(pool=pool, z_mat=test_z_mat, network_model_class=network_model_class,
#                                              network_model_class_kwargs=network_model_class_kwargs,
#                                              network_net_size=network_net_size, number_of_nets=number_of_nets)
#
#     p_value = sum(trial_test_statistic < test_statistic_sample_vet) / number_of_nets
#     print(f"Scenario: {scenario} Sample size: {sample_size} trial: {trial_index} is done. p-value: {p_value}")
#     return p_value


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


def naive_chisq_method(trial_index, scenario, data_directory_name, sample_size):
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

    :param x_y_mat: An n x 2 numpy array. Each row is the response of the ith observation. First column corresponds to
        x.
    :param z_mat: A 2D numpy array. Each row is a sample
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


def stratified_chisq_method(trial_index, scenario, data_directory_name, sample_size, cluster_number=2):
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
        1. x: An n x 1 numpy array corresponding to X.
        2. y: An n x 1 numpy array corresponding to Y.
    """
    x = x_y_mat[:, 0][:, np.newaxis]
    y = x_y_mat[:, 1][:, np.newaxis]
    return x, y


def ccit_method(trial_index, scenario, data_directory_name, sample_size, **kwargs):
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
