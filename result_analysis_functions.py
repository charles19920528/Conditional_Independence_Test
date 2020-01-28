import multiprocessing as mp
from functools import partial
import os
# Only run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import generate_train_fucntions as gt
import hyperparameters as hp

#####################################
# Get test statistic for one trail. #
#####################################
# Method specific functions to bbtain test statistic for one trial.
def ising_test_statistic_one_trial(trail_index, one_sample_size_result_dict):
    """
    Calculate the average of squared Jxy in one trail.

    :param trail_index: An integer indicating the index of the simulation trail of which we extract the test statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    a particular sample size.

    :return:
    jxy_squared_mean: An scalar which is the average of squared Jxy..
    """
    jxy_squared_vet = np.square(one_sample_size_result_dict[trail_index][:, 2])
    jxy_squared_mean = np.mean(jxy_squared_vet)

    return jxy_squared_mean


def naive_sq_statistic_one_trail(trail_index, one_sample_size_result_dict, isPvalue):
    """
    Obatin either p-value or Chi squared statistic of the Naive Chi square test.

    :param trail_index: An integer indicating the index of the simulation trail of which we extract the test statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    a particular sample size.

    :return:
    A scalar. If isPvalue is true, return a scalar between 0 and 1. Otherwise, return a postive scalar.
    """
    if isPvalue:
        pvalue = one_sample_size_result_dict[trail_index][1]
        return pvalue
    else:
        chisq_statistic = one_sample_size_result_dict[trail_index][0]
        return chisq_statistic


def stratified_sq_statistic_one_trail(trail_index, one_sample_size_result_dict):
    """
    Obatin the sum of the test statistic of the stratified Chi Squared test.

    :param trail_index: An integer indicating the index of the simulation trail of which we extract the test statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    a particular sample size.

    :return:
    test_statisitc: A positive scalar.
    """
    test_statisitc = one_sample_size_result_dict[trail_index]
    return test_statisitc


def ccit_one_trail(trail_index, one_sample_size_result_dict):
    """
    Obatin the sum of the test statistic of the CCIT test.

    :param trail_index: An integer indicating the index of the simulation trail of which we extract the test statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    a particular sample size.

    :return:
    test_statisitc: A positive scalar between 0 and 1 representing the percentage of samples which are correctly
    classified.
    """
    test_statisitc = 1 - one_sample_size_result_dict[trail_index]
    return test_statisitc


def ising_residual_statistic_one_trail(trail_index, sample_size,  scenario, data_directory_name,
                                       one_sample_size_result_dict):
    """
    Obatin the residual test statistic of the Ising model for one trail.

    :param: sample_size:
    :param trail_index: An integer indicating the index of the simulation trail of which we extract the test
    statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model
    for a particular sample size.

    :return:
    test_statisitc: A positive scalar between 0 and 1 representing the percentage of samples which are correctly
    classified.
    """
    parameter_mat = one_sample_size_result_dict[trail_index]
    pmf_mat = gt.pmf_null(x = 1, hx = parameter_mat)
    expectation_mat = pmf_mat - (1 - pmf_mat)
    expectation_x_vet = expectation_mat[:, 0]
    expectation_y_vet = expectation_mat[:, 1]

    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trail_index}.txt",
                         dtype=np.float32)
    centered_x_vet = x_y_mat[:, 0] - expectation_x_vet
    centered_y_vet = x_y_mat[:, 1] - expectation_y_vet
    r_vet = centered_x_vet * centered_y_vet

    test_statistic_numerator = np.sqrt(sample_size) * np.mean(r_vet)
    test_statistic_denominator = np.std(r_vet)
    test_statistic = np.abs(test_statistic_numerator / test_statistic_denominator)

    return test_statistic


################################################################
# Get test statistic for all trails with the same sample size. #
################################################################
# Shared functions to obtain fpr, tpr.
def test_statistic_one_sample_size(one_sample_size_null_result_dict, one_sample_size_alt_result_dict, number_of_trails,
                                   test_statistic_one_trail, **kwargs):
    """
    Apply test_statistic_one_trail function to each result dictionary of a particular sample size to obtain test
    statistics for all trails.

    :param one_sample_size_null_result_dict: A dictionary containing the raw outputs simulated under the null
    given a particular sample size.
    :param one_sample_size_alt_result_dict: A dictionary containing the raw outputs simulated under the
    alternative given a particular sample size.
    :param number_of_trails:
    :param test_statistic_one_trail: A function which extract the test statistic for one trail. It should be one of the
    functions defnied above. 
    :param kwargs: Other named arguments to be passed into the test_statistic_one_trail function. 

    :return:
    """
    pool = mp.Pool(hp.process_number)
    trail_index_vet = range(number_of_trails)

    if test_statistic_one_trail == ising_residual_statistic_one_trail:
        null_test_statistic_vet_one_sample = pool.map(partial(ising_residual_statistic_one_trail,
                                                              one_sample_size_result_dict=
                                                              one_sample_size_null_result_dict, scenario = "null",
                                                              **kwargs), trail_index_vet)
        alt_test_statistic_vet_one_sample = pool.map(partial(ising_residual_statistic_one_trail,
                                                             one_sample_size_result_dict=
                                                             one_sample_size_alt_result_dict, scenario = "alt",
                                                             **kwargs), trail_index_vet)
    else:
        null_test_statistic_vet_one_sample = pool.map(partial(test_statistic_one_trail,
                                                              one_sample_size_result_dict=
                                                              one_sample_size_null_result_dict, **kwargs),
                                                      trail_index_vet)
        alt_test_statistic_vet_one_sample = pool.map(partial(test_statistic_one_trail,
                                                             one_sample_size_result_dict=
                                                             one_sample_size_alt_result_dict, **kwargs,),
                                                     trail_index_vet)

    return (null_test_statistic_vet_one_sample, alt_test_statistic_vet_one_sample)


#######################
# Compute fpr and tpr #
#######################
def fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, number_of_trails):
    """
    A wrap up function is used in the fpr_tpr function which usess the multiprocessing Pool function.

    :param test_statistic_one_sample_size_tuple:
    :param number_of_trails: An integer which is the number of trails we simulate for each sample size

    :return:
    Two ists containing false positive rates and true positive rates which can be used to draw the RoC curve.
    """
    true_label = np.repeat([-1, 1], np.repeat(number_of_trails, 2))
    combined_test_statistic_vet = np.concatenate(test_statistic_one_sample_size_tuple)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, combined_test_statistic_vet, pos_label=1)

    return fpr, tpr


def fpr_tpr(null_result_dict, alt_result_dict, test_statistic_one_trail, number_of_trails = hp.number_of_trails,
            **kwargs):
    """
    A wrapper function which compute fpr and tpr for simulations of different samples sizes.

    :param null_result_dict:
    :param alt_result_dict:
    :param test_statistic_one_trail:
    :param number_of_trails:
    :param kwargs:

    :return:
    fpr_tpr_dict: A dictionary containing lists of fpr and tpr of different sample sizes.
    """
    fpr_tpr_dict = dict()
    for sample_size in null_result_dict.keys():
        one_sample_size_null_result_dict = null_result_dict[sample_size]
        one_sample_size_alt_result_dict = alt_result_dict[sample_size]

        if test_statistic_one_trail == ising_residual_statistic_one_trail:
            test_statistic_one_sample_size_tuple = test_statistic_one_sample_size(one_sample_size_null_result_dict=
                                                                                  one_sample_size_null_result_dict,
                                                                                  one_sample_size_alt_result_dict=
                                                                                  one_sample_size_alt_result_dict,
                                                                                  number_of_trails=number_of_trails,
                                                                                  test_statistic_one_trail=
                                                                                  test_statistic_one_trail,
            sample_size = sample_size,**kwargs)
        else:
            test_statistic_one_sample_size_tuple = test_statistic_one_sample_size(one_sample_size_null_result_dict=
                                                                                  one_sample_size_null_result_dict,
                                                                                  one_sample_size_alt_result_dict=
                                                                                  one_sample_size_alt_result_dict,
                                                                                  number_of_trails=number_of_trails,
                                                                                  test_statistic_one_trail=
                                                                                  test_statistic_one_trail, **kwargs)

        fpr, tpr = fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, number_of_trails)

        fpr_tpr_dict[sample_size] = [fpr, tpr]

    return fpr_tpr_dict


#############
# PLot Roce #
#############
def plot_roc(fpr_tpr_dict, model_for_main_title, result_directory_name):
    """
    Assuming there are only four sample size we are simulating. We plot the RoC curve and save the plot under
    ./results/plots/{model_for_main_title}.png.
    
    :param fpr_tpr_dict: A dictionary which is the output of the fpr_tpr function.
    :param model_for_main_title: A string ('str' class) which we use to name the graph to save as 
    {model_for_main_title}.png.
    :param result_directory_name: A string ('str' class) which is the name of the directory to store the plot.
    
    :return:
    None
    """
    fig, ax = plt.subplots(2, 2)
    sample_size = 30
    ax[0, 0].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    ax[0, 0].axvline(x = 0.05, color = "red")
    ax[0, 0].set_title(f"Sample size {sample_size}")

    sample_size = 100
    ax[0, 1].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    ax[0, 1].axvline(x = 0.05, color = "red")
    ax[0, 1].set_title(f"Sample size {sample_size}")

    sample_size = 500
    ax[1, 0].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    ax[1, 0].axvline(x = 0.05, color = "red")
    ax[1, 0].set_title(f"Sample size {sample_size}")

    sample_size = 1000
    ax[1, 1].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    ax[1, 1].axvline(x = 0.05, color = "red")
    ax[1, 1].set_title(f"Sample size {sample_size}")

    fig.suptitle(f"RoC Curves of {model_for_main_title}")
    fig.show()
    fig.savefig(f"./results/plots/{result_directory_name}/{model_for_main_title}.png")