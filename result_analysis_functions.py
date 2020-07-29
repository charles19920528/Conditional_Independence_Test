from functools import partial
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import hyperparameters as hp

# Only run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#####################################
# Get test statistic for one trial. #
#####################################
# Method specific functions to obtain test statistic for one trial.
def ising_test_statistic_one_trial(trial_index, one_sample_size_result_dict):
    """
    Extract the test statistic computed based on data of {trial_index}th trial.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.

    :return:
        A scalar which is the test statistics of the neural Ising model.
    """

    return one_sample_size_result_dict[trial_index]["train_test_statistic"]


def naive_sq_statistic_one_trial(trial_index, one_sample_size_result_dict, isPvalue):
    """
    Obtain either p-value or Chi squared statistic of the Naive Chi square test based on data of {trial_index}th trial.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.
    :param isPvalue: A boolean. If true, return p-value. Otherwise, return the Chi-squared test statistic.

    :return:
        A scalar.
    """
    if isPvalue:
        pvalue = one_sample_size_result_dict[trial_index][1]
        return pvalue
    else:
        chisq_statistic = one_sample_size_result_dict[trial_index][0]
        return chisq_statistic


def stratified_sq_statistic_one_trial(trial_index, one_sample_size_result_dict):
    """
    Obtain the sum of the test statistic of the stratified Chi Squared test based on data of {trial_index}th trial.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.

    :return:
        A scalar.
    """
    test_statistic = one_sample_size_result_dict[trial_index]
    return test_statistic


def ccit_one_trial(trial_index, one_sample_size_result_dict):
    """
    Obatin the sum of the test statistic of the CCIT test based on data of {trial_index}th trial.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.

    :return:
        test_statistic: A non-negative scalar between 0 and 1 representing the percentage of samples which are correctly
        classified.
    """
    test_statistic = 1 - one_sample_size_result_dict[trial_index]
    return test_statistic


# Not in use now.
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
                                   trial_index_vet, test_statistic_one_trial, **kwargs):
    """
    Use the multiprocess Pool class to call test_statistic_one_trial function on each result dictionary of a particular
        sample size to obtain the test statistics for all trials.

    :param pool: A multiprocessing.pool.Pool instance.
    :param one_sample_size_null_result_dict: A dictionary containing the raw output of sf.(ising_)simulation_loop
        function under the null given a particular sample size.
    :param one_sample_size_alt_result_dict: A dictionary containing the raw output of sf.(ising_)simulation_loop
        function under the alternative given a particular sample size.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.
    :param test_statistic_one_trial: A function which extract the test statistic for one trial. It should be one of the
        functions defined above.
    :param kwargs: Additional keyword arguments to pass into the test_statistic_one_trial function if necessary.

    :return:
        1. null_test_statistic_vet_one_sample_vet: An array of scalars which are test statistics computed based on
            trials which are of the sample size and generated under the null.
        2. alt_test_statistic_vet_one_sample_vet: An array of scalars which are test statistics computed based on
            trials which are of the sample size and generated under the alt.
    """
    null_test_statistic_vet_one_sample_vet = pool.map(partial(test_statistic_one_trial,
                                                              one_sample_size_result_dict=
                                                              one_sample_size_null_result_dict, **kwargs),
                                                      trial_index_vet)
    alt_test_statistic_vet_one_sample_vet = pool.map(partial(test_statistic_one_trial,
                                                             one_sample_size_result_dict=
                                                             one_sample_size_alt_result_dict, **kwargs),
                                                     trial_index_vet)

    return null_test_statistic_vet_one_sample_vet, alt_test_statistic_vet_one_sample_vet


#######################
# Compute fpr and tpr #
#######################
def fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, trial_index_vet):
    """
    A wrapper function which is used in the fpr_tpr function which uses the multiprocessing Pool function.

    :param test_statistic_one_sample_size_tuple: A tuple of outputs of the test_statistic_one_sample_size function.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.

    :return:
        Two arrays containing false positive rates and true positive rates which can be used to draw the RoC curve.
    """
    number_of_trials = len(trial_index_vet)

    true_label = np.repeat([-1, 1], np.repeat(number_of_trials, 2))
    combined_test_statistic_vet = np.concatenate(test_statistic_one_sample_size_tuple)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, combined_test_statistic_vet, pos_label=1)

    return fpr, tpr


def fpr_tpr(pool, null_result_dict, alt_result_dict, test_statistic_one_trial, trial_index_vet,
            **kwargs):
    """
    A wrapper function which compute fpr and tpr for simulations of all samples sizes.

    :param pool: A multiprocessing.pool.Pool instance.
    :param null_result_dict: A dictionary. It should be an output of the sf.(ising_)simulation_loop function called on
        null samples.
    :param alt_result_dict: A dictionary. It should be an output of the sf.(ising_)simulation_loop function called on
        alternative samples.
    :param test_statistic_one_sample_size_tuple: A tuple of outputs of the test_statistic_one_sample_size function.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.
    :param kwargs: Additional keyword arguments to pass into the test_statistic_one_trial function if necessary.

    :return:
        fpr_tpr_dict: A dictionary containing lists of fpr and tpr of all sample sizes.
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
                                                                              trial_index_vet=trial_index_vet,
                                                                              test_statistic_one_trial=
                                                                              test_statistic_one_trial, **kwargs)

        fpr, tpr = fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, trial_index_vet)

        fpr_tpr_dict[sample_size] = [fpr, tpr]

    return fpr_tpr_dict


#############
# PLot Roce #
#############
def plot_roc(fpr_tpr_dict, title, result_directory_name):
    """
    Assuming there are only four sample size we are simulating. We plot the RoC curve and save the plot under the path
    ./results/plots/{result_directory_name} with name {model_for_main_title}.png.
    
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
