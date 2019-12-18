import numpy as np
from scipy.stats import chi2_contingency
import hyperparameters as hp
import pandas as pd
from functools import partial
import multiprocessing as mp
import time
import pickle


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

def simulation_wrapper_naive(simulation_index, scenario, sample_size):
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

if __name__ == "__main__":

    #######################
    # Test under the null #
    #######################
    naive_chisq_result_null_dict = dict()
    # start_time = time.time()
    for sample_size in hp.sample_size_vet:
        pool = mp.Pool(processes=hp.process_number)
        simulation_index_vet = range(hp.simulation_times)
        pool_result_vet = pool.map(partial(simulation_wrapper_naive, sample_size=sample_size, scenario="null"),
                                   simulation_index_vet)

        sample_result_dict = dict(pool_result_vet)
        naive_chisq_result_null_dict[sample_size] = sample_result_dict
    # print("--- %s seconds ---" % (time.time() - start_time))
    with open("./results/naive_chisq_result_null_dict.p", "wb") as fp:
        pickle.dump(naive_chisq_result_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    ##############################
    # Test under the alternative #
    ##############################
    naive_chisq_result_alt_dict = dict()
    # start_time = time.time()
    for sample_size in hp.sample_size_vet:
        pool_result_vet = pool.map(partial(simulation_wrapper_naive, sample_size=sample_size, scenario="alt"),
                                   simulation_index_vet)

        sample_result_dict = dict(pool_result_vet)
        naive_chisq_result_alt_dict[sample_size] = sample_result_dict
    # print("--- %s seconds ---" % (time.time() - start_time))
    with open("./results/naive_chisq_result_alt_dict.p", "wb") as fp:
        pickle.dump(naive_chisq_result_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    """
    Sequential version. Saved for debugging.
    #######################
    # Test under the null #
    #######################
    start_time = time.time()
    naive_chisq_result_null_dict = dict()
    for sample_size in hp.sample_size_vet:

        sample_result_dict = sample_result_dict = dict()
        for simulation in range(hp.simulation_times):
            x_y_mat = np.loadtxt(f"./data/null/x_y_mat_{sample_size}_{simulation}.txt")
            sample_result_dict[simulation] = chi_squared_test(x_y_mat)

        naive_chisq_result_null_dict[sample_size] = sample_result_dict
    print("--- %s seconds ---" % (time.time() - start_time))

    ##############################
    # Test under the alternative #
    ##############################
    naive_chisq_result_alt_dict = dict()
    for sample_size in hp.sample_size_vet:

        sample_result_dict = dict()
        for simulation in range(hp.simulation_times):
            x_y_mat = np.loadtxt(f"./data/alt/x_y_mat_{sample_size}_{simulation}.txt")
            sample_result_dict[simulation] = chi_squared_test(x_y_mat)

        naive_chisq_result_alt_dict[sample_size] = sample_result_dict
    """


