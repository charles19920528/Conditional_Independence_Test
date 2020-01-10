import multiprocessing as mp
from functools import partial
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from CCIT import CCIT

import generate_train_fucntions as gt
import hyperparameters as hp

####################
# Simulation loops #
####################
def simulation_loop(simulation_wrapper, scenario, result_dict_name, sample_size_vet = hp.sample_size_vet,
                    simulation_times = hp.simulation_times, process_number = hp.process_number):
    result_dict = dict()
    for sample_size in sample_size_vet:
        pool = mp.Pool(processes=process_number)
        simulation_index_vet = range(simulation_times)

        pool_result_vet = pool.map(partial(simulation_wrapper, sample_size=sample_size, scenario=scenario),
                                   simulation_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

        with open(f"./results/{result_dict_name}_result_{scenario}_dict.p", "wb") as fp:
            pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{scenario}, {sample_size} finished")


def oracle_ising_simulation_loop(scenario, result_dict_name, sample_size_vet = hp.sample_size_vet,
                                 epoch_vet = hp.epoch_vet, simulation_times = hp.simulation_times,
                                 process_number = hp.process_number):
    result_dict = dict()
    for sample_size, epoch in zip(sample_size_vet, epoch_vet):
        pool = mp.Pool(processes=process_number)
        simulation_index_vet = range(simulation_times)

        pool_result_vet = pool.map(partial(ising_simulation_wrapper, sample_size=sample_size, scenario=scenario,
                                           epoch=epoch), simulation_index_vet)

        result_dict[sample_size] = dict(pool_result_vet)

        with open(f"./results/{result_dict_name}_result_{scenario}_dict.p", "wb") as fp:
            pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{scenario}, {sample_size} finished")


#####################
# Wrapper functions #
#####################
# Ising simulation
def ising_simulation_wrapper(simulation_index, scenario, sample_size, epoch):
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt", dtype = np.float32)
    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}_{simulation_index}.txt", dtype = np.float32)

    ising_training_pool_instance = gt.IsingTrainingPool(z_mat=z_mat, epoch = epoch)

    predicted_parameter_mat = ising_training_pool_instance.trainning(x_y_mat = x_y_mat)

    print(f"{scenario}: Sample size {sample_size} simulation {simulation_index} is done.")

    return (simulation_index, predicted_parameter_mat)


# Naive Chisq
def chi_squared_test(x_y_mat):
    """
    Perform unconditional Chi-squared test.

    :param x_y_mat: A 2-d numpy array with columns corresponding to x and y.
    :return: result_vet: A length 2 tuple. result_vet[0] is the test statistic and result_vet[1] is the p-value.
    """
    x_vet = x_y_mat[:, 0]
    y_vet = x_y_mat[:, 1]
    contingency_table = pd.crosstab(x_vet, y_vet, rownames = "x", colnames = "y")

    result_vet = chi2_contingency(contingency_table)[0:2]
    return result_vet


def naive_chisq_wrapper(simulation_index, scenario, sample_size):
    """
    A wrapper function for the multiprocessing Pool function. It will be passed into the partial function.
    The pool function will run iteration in parallel given a sample size and a scenario.
    This function perform Chi squared test on {simulation_index}th sample with sample size
    {sample_size} under the {scenario} hypothesis.
    The return will be used to create a dictionary.

    :param simulation_index: An integer indicating {simulation_index}th simulated.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
    under the null or alternative hypothesis.
    :param sample_size: An integer.
    :return: A tuple (simulation_index, result_vet). result_vet is the return of the chi_squared_test function.
    """
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt")
    result_vet = chi_squared_test(x_y_mat)
    return (simulation_index, result_vet)


# Stratified Chisq
def stratify_x_y_mat(x_y_mat, z_mat, cluster_number = 2):
    """
    Cluster data into {cluster_number} of clusters.
    :param x_y_mat: An n x 2 numpy array. Each row is the response of the ith observation.
    First column corresponds to x.
    :param z_mat: A 2D numpy array. Each row is a data point.
    :param cluster_number: An integer which is the number of clusters to form by the KMeans function.
    :return: z_mat_vet: A python list of length {cluster_number}. Each element is a 2D numpy array of a data cluster.
    """
    kmeans_model = KMeans(n_clusters=cluster_number, random_state=0)
    kmeans_result_vet = kmeans_model.fit(z_mat).labels_
    x_y_mat_vet = [x_y_mat[kmeans_result_vet == i, :] for i in range(cluster_number)]

    return x_y_mat_vet


def stratified_chi_squared_test(x_y_mat_vet):
    """
    Compute test statistic.

    :param x_y_mat_vet: A python list which is the output of the stratify_x_y_mat function.
    :return: A numeric which is the sum of of the chi-squared statistic on each stratum.
    """
    chi_square_statistic_vet = np.zeros(len(x_y_mat_vet))
    for iteration, x_y_mat in enumerate(x_y_mat_vet):
        chi_square_statistic_vet[iteration] = chi_squared_test(x_y_mat)[0]

    return sum(chi_square_statistic_vet)


def stratified_chisq_wrapper(simulation_index, scenario, sample_size, cluster_number= 2):
    """
    A wrapper function for the multiprocessing Pool function. It will be passed into the partial function.
    The pool function will run iteration in parallel given a sample size and a scenario.
    This function perform Chi squared test on {simulation_index}th sample with sample size
    {sample_size} under the {scenario} hypothesis.
    The return will be used to create a dictionary.

    :param simulation_index: An integer indicating {simulation_index}th simulated.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
    under the null or alternative hypothesis.
    :param sample_size: An integer.
    :return: A tuple (simulation_index, result_vet). result_vet is the return of the chi_squared_test function.
    """
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt")
    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}_{simulation_index}.txt", dtype=np.float32)

    x_y_mat_vet = stratify_x_y_mat(x_y_mat = x_y_mat, z_mat = z_mat, cluster_number = cluster_number)
    test_statistic = stratified_chi_squared_test(x_y_mat_vet)

    return (simulation_index, test_statistic)


# CCIT
def process_x_y_mat(x_y_mat):
    x = x_y_mat[:, 0][:, np.newaxis]
    y = x_y_mat[:, 1][:, np.newaxis]
    return x, y


def ccit_wrapper(simulation_index, scenario, sample_size, **kwargs):
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt")
    x_array, y_array = process_x_y_mat(x_y_mat)
    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}_{simulation_index}.txt", dtype=np.float32)

    p_value = CCIT.CCIT(x_array, y_array, z_mat, **kwargs)

    print(f"{scenario}: Sample size {sample_size} simulation {simulation_index} is done.")

    return (simulation_index, p_value)


##############
# Simulation #
##############
# Naive Chisq simulation
simulation_loop(simulation_wrapper = naive_chisq_wrapper,  scenario = "null", result_dict_name = "naive_chisq")
simulation_loop(simulation_wrapper = naive_chisq_wrapper,  scenario = "alt", result_dict_name = "naive_chisq")

# Stratified Chisq simulation
simulation_loop(simulation_wrapper = stratified_chisq_wrapper, scenario = "null",
                   result_dict_name = "stratified_chisq")
simulation_loop(simulation_wrapper = stratified_chisq_wrapper, scenario = "alt",
                   result_dict_name = "stratified_chisq")

# Ising simulation
start_time = time.time()

oracle_ising_simulation_loop(scenario = "null", result_dict_name = "ising")
oracle_ising_simulation_loop(scenario = "alt", result_dict_name = "ising")

print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))

# CCIT simulation
process_number_ccit = 10
start_time = time.time()

simulation_loop(simulation_wrapper = ccit_wrapper, scenario = "null", result_dict_name = "ccit",
                process = process_number_ccit)
simulation_loop(simulation_wrapper = ccit_wrapper, scenario = "alt", result_dict_name = "ccit",
                process = process_number_ccit)

print("CCIT simulation takes %s seconds to finish." % (time.time() - start_time))

