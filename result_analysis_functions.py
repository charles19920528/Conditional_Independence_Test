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


##########################################################
# Get sampling distribution of the Ising test statistic. #
##########################################################
def ising_test_statistic_distribution_one_trial(z_mat, network_model_class, network_model_class_kwargs,
                                                network_net_size=500):

    network_net_vet = [network_model_class(network_model_class_kwargs) for i in range(network_net_size)]






#####################################
# Get test statistic for one trial. #
#####################################
def ising_pvalue_one_trial(trial_index, one_sample_size_result_dict):
    pass

# Method specific functions to obtain test statistic for one trial.
def ising_test_statistic_one_trial(trial_index, one_sample_size_result_dict):
    """

    :param trial_index:
    :param one_sample_size_result_dict:
    :return:
    """
    # """
    # Calculate the average of squared Jxy in one trial.
    #
    # :param trial_index: An integer indicating the index of the simulation trial of which we extract the test statistic.
    # :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    # a particular sample size.
    #
    # :return:
    # jxy_squared_mean: An scalar which is the average of squared Jxy..
    # """

    # jxy_squared_vet = np.square(one_sample_size_result_dict[trial_index][:, 2])
    # jxy_squared_mean = np.mean(jxy_squared_vet)


    # return jxy_squared_mean
    return one_sample_size_result_dict[trial_index]["test_statistic"]


def naive_sq_statistic_one_trial(trial_index, one_sample_size_result_dict, isPvalue):
    """
    Obtain either p-value or Chi squared statistic of the Naive Chi square test.

    :param trial_index: An integer indicating the index of the simulation trial of which we extract the test statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    a particular sample size.

    :return:
    A scalar. If isPvalue is true, return a scalar between 0 and 1. Otherwise, return a postive scalar.
    """
    if isPvalue:
        pvalue = one_sample_size_result_dict[trial_index][1]
        return pvalue
    else:
        chisq_statistic = one_sample_size_result_dict[trial_index][0]
        return chisq_statistic


def stratified_sq_statistic_one_trial(trial_index, one_sample_size_result_dict):
    """
    Obtain the sum of the test statistic of the stratified Chi Squared test.

    :param trial_index: An integer indicating the index of the simulation trial of which we extract the test statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    a particular sample size.

    :return:
    test_statistic: A positive scalar.
    """
    test_statistic = one_sample_size_result_dict[trial_index]
    return test_statistic


def ccit_one_trial(trial_index, one_sample_size_result_dict):
    """
    Obatin the sum of the test statistic of the CCIT test.

    :param trial_index: An integer indicating the index of the simulation trial of which we extract the test statistic.
    :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model for
    a particular sample size.

    :return:
    test_statistic: A positive scalar between 0 and 1 representing the percentage of samples which are correctly
    classified.
    """
    test_statistic = 1 - one_sample_size_result_dict[trial_index]
    return test_statistic


# def ising_residual_statistic_one_trial(trial_index, sample_size, scenario, data_directory_name,
#                                        one_sample_size_result_dict):
#     """
#     Obatin the residual test statistic of the Ising model for one trial.
#
#     :param trial_index: An integer indicating the index of the simulation trial of which we extract the test
#     statistic.
#     :param sample_size: An integer which is the sample size corresponding to the one_sample_size_result_dict.
#     :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
#     under the null or alternative hypothesis.
#     :param data_directory_name: A string ('str' class) of the path towards the simulation data.
#     :param one_sample_size_result_dict: A dictionary which contains all results of the simulation of the Ising model
#     for a particular sample size.
#
#     :return:
#     test_statistic: A positive scalar between 0 and 1 representing the percentage of samples which are correctly
#     classified.
#     """
#     parameter_mat = one_sample_size_result_dict[trial_index]
#     pmf_mat = gt.pmf_null(x=1, hx=parameter_mat)
#     expectation_mat = pmf_mat - (1 - pmf_mat)
#     expectation_x_vet = expectation_mat[:, 0]
#     expectation_y_vet = expectation_mat[:, 1]
#
#     x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt",
#                          dtype=np.float32)
#     centered_x_vet = x_y_mat[:, 0] - expectation_x_vet
#     centered_y_vet = x_y_mat[:, 1] - expectation_y_vet
#     r_vet = centered_x_vet * centered_y_vet
#
#     test_statistic_numerator = np.sqrt(sample_size) * np.mean(r_vet)
#     test_statistic_denominator = np.std(r_vet)
#     test_statistic = np.abs(test_statistic_numerator / test_statistic_denominator)
#
#     return test_statistic


################################################################
# Get test statistic for all trials with the same sample size. #
################################################################
# Shared functions to obtain fpr, tpr.
def test_statistic_one_sample_size(pool, one_sample_size_null_result_dict, one_sample_size_alt_result_dict,
                                   number_of_trials, test_statistic_one_trial, **kwargs):
    """
    Apply test_statistic_one_trial function to each result dictionary of a particular sample size to obtain test
    statistics for all trials.

    :param
    :param one_sample_size_null_result_dict: A dictionary containing the raw outputs simulated under the null
    given a particular sample size.
    :param one_sample_size_alt_result_dict: A dictionary containing the raw outputs simulated under the
    alternative given a particular sample size.
    :param number_of_trials:
    :param test_statistic_one_trial: A function which extract the test statistic for one trial. It should be one of the
    functions defined above.
    :param kwargs: Other named arguments to be passed into the test_statistic_one_trial function.

    :return:
    """
    trial_index_vet = range(number_of_trials)



    null_test_statistic_vet_one_sample = pool.map(partial(test_statistic_one_trial,
                                                          one_sample_size_result_dict=
                                                          one_sample_size_null_result_dict, **kwargs),
                                                  trial_index_vet)
    alt_test_statistic_vet_one_sample = pool.map(partial(test_statistic_one_trial,
                                                         one_sample_size_result_dict=
                                                         one_sample_size_alt_result_dict, **kwargs, ),
                                                     trial_index_vet)

    return null_test_statistic_vet_one_sample, alt_test_statistic_vet_one_sample


#######################
# Compute fpr and tpr #
#######################
def fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, number_of_trials):
    """
    A wrap up function is used in the fpr_tpr function which uses the multiprocessing Pool function.

    :param test_statistic_one_sample_size_tuple:
    :param number_of_trials: An integer which is the number of trials we simulate for each sample size

    :return:
    Two ists containing false positive rates and true positive rates which can be used to draw the RoC curve.
    """
    true_label = np.repeat([-1, 1], np.repeat(number_of_trials, 2))
    combined_test_statistic_vet = np.concatenate(test_statistic_one_sample_size_tuple)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, combined_test_statistic_vet, pos_label=1)

    return fpr, tpr


def fpr_tpr(pool, null_result_dict, alt_result_dict, test_statistic_one_trial, number_of_trials=hp.number_of_trials,
            **kwargs):
    """
    A wrapper function which compute fpr and tpr for simulations of different samples sizes.

    :param pool:
    :param null_result_dict:
    :param alt_result_dict:
    :param test_statistic_one_trial:
    :param number_of_trials:
    :param kwargs:

    :return:
    fpr_tpr_dict: A dictionary containing lists of fpr and tpr of different sample sizes.
    """
    fpr_tpr_dict = dict()
    for sample_size in null_result_dict.keys():
        one_sample_size_null_result_dict = null_result_dict[sample_size]
        one_sample_size_alt_result_dict = alt_result_dict[sample_size]

        test_statistic_one_sample_size_tuple = test_statistic_one_sample_size(pool=pool,
                                                                              one_sample_size_null_result_dict=
                                                                              one_sample_size_null_result_dict,
                                                                              one_sample_size_alt_result_dict=
                                                                              one_sample_size_alt_result_dict,
                                                                              number_of_trials=number_of_trials,
                                                                              test_statistic_one_trial=
                                                                              test_statistic_one_trial, **kwargs)

        fpr, tpr = fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, number_of_trials)

        fpr_tpr_dict[sample_size] = [fpr, tpr]

    return fpr_tpr_dict


#############
# PLot Roce #
#############
def plot_roc(fpr_tpr_dict, title, result_directory_name):
    """
    Assuming there are only four sample size we are simulating. We plot the RoC curve and save the plot under
    ./results/plots/{model_for_main_title}.png.
    
    :param fpr_tpr_dict: A dictionary which is the output of the fpr_tpr function.
    :param title: A string ('str' class) which we use to name the graph and to save the image as
    {title}.png.
    :param result_directory_name: A string ('str' class) which is the name of the directory to store the plot.
    
    :return:
    None
    """
    fig, ax = plt.subplots(2, 2)
    sample_size = 50
    ax[0, 0].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
#    ax[0, 0].axvline(x=0.05, color="red")
    ax[0, 0].set_title(f"Sample size {sample_size}")

    sample_size = 100
    ax[0, 1].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
#    ax[0, 1].axvline(x=0.05, color="red")
    ax[0, 1].set_title(f"Sample size {sample_size}")

    sample_size = 500
    ax[1, 0].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
#    ax[1, 0].axvline(x=0.05, color="red")
    ax[1, 0].set_title(f"Sample size {sample_size}")

    sample_size = 1000
    ax[1, 1].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
#    ax[1, 1].axvline(x=0.05, color="red")
    ax[1, 1].set_title(f"Sample size {sample_size}")

    fig.suptitle(f"RoC Curves of {title}")
    fig.show()
    fig.savefig(f"./results/plots/{result_directory_name}/{title}.png")
