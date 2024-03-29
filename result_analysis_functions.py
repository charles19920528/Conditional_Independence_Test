from functools import partial
import os
import pickle
import numpy as np
import statsmodels.api as sm
import scipy.stats.distributions as dist
from scipy.stats import chi2
from scipy.optimize import fsolve
from sklearn import metrics
import matplotlib.pyplot as plt
import generate_train_functions as gt
import test_statistic as ts
import hyperparameters as hp

# Only run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#####################################
# Get test statistic for one trial. #
#####################################
# def ising_score_test_statistic_one_trial(trial_index, one_sample_size_result_dict, sample_size, scenario,
#                                         data_directory_name, network_test_args_dict_dict):
#     test_indices_vet = one_sample_size_result_dict[trial_index]["test_indices_vet"]
#     train_indices_vet = np.arange(sample_size)[~np.in1d(np.arange(sample_size), test_indices_vet)]
#     train_indices_boolean = False
#     if train_indices_boolean:
#         indices_vet = train_indices_vet
#     else:
#         indices_vet = test_indices_vet
#
#     # predicted_parameter_mat = one_sample_size_result_dict[trial_index]["predicted_parameter_mat"][test_indices_vet, :]
#     network_weights_vet = one_sample_size_result_dict[trial_index]["network_weights_vet"]
#
#     x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt")
#     x_y_mat = x_y_mat[indices_vet, :]
#
#     z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt")
#     z_mat = z_mat[indices_vet, :]
#
#     test_statistic = ts.score_test(x_y_mat=x_y_mat, z_mat=z_mat,
#                                   network_test_args_dict=network_test_args_dict_dict[scenario],
#                                   network_weights_vet=network_weights_vet)
#
#     return test_statistic


def ising_score_test_statistic_one_trial(trial_index, one_sample_size_result_dict, sample_size, scenario,
                                        data_directory_name, sandwich_boolean, n_batches=None, batch_size=None):
    test_indices_vet = one_sample_size_result_dict[trial_index]["test_indices_vet"]
    if test_indices_vet is None:
        indices_vet = np.arange(sample_size)
    else:
        indices_vet = test_indices_vet

    network_weights_vet = one_sample_size_result_dict[trial_index]["network_weights_vet"]
    j_mat = one_sample_size_result_dict[trial_index]["predicted_parameter_mat"][indices_vet, :]
    final_linear_input_mat = one_sample_size_result_dict[trial_index]["final_linear_input_mat"][indices_vet, :]

    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt")
    x_y_mat = x_y_mat[indices_vet, :]

    score_test_instance = ts.ScoreTest(x_y_mat=x_y_mat, j_mat=j_mat, final_linear_input_mat=final_linear_input_mat,
                                       network_weights_vet=network_weights_vet, sandwich_boolean=sandwich_boolean)
    if n_batches is None or batch_size is None:
        test_statistic = score_test_instance.get_test_statistic()
    else:
        test_statistic = score_test_instance.p_value(n_batches=n_batches, batch_size=batch_size)
        print(f"{scenario}, {sample_size}, trial: {trial_index}, p_value: {test_statistic}.")

    return test_statistic


def ising_sq_statistic_one_trial(trial_index, one_sample_size_result_dict, jxy_boolean):
    test_indices_vet = one_sample_size_result_dict[trial_index]["test_indices_vet"]
    predicted_parameter_mat = one_sample_size_result_dict[trial_index]["predicted_parameter_mat"][test_indices_vet, :]
    network_weights_vet = one_sample_size_result_dict[trial_index]["network_weights_vet"]

    if jxy_boolean:
        test_statistics = np.sum(predicted_parameter_mat[: 2] ** 2)
    else:
        weights_vet = np.concatenate([network_weights_vet[-2][:, 2], [network_weights_vet[-1][-1]]])
        test_statistics = np.sum(weights_vet ** 2)
    return test_statistics


# Method specific functions to obtain test statistic for one trial.
def ising_powerful_test_statistic_one_trial(trial_index, one_sample_size_result_dict, sample_size, scenario,
                                            data_directory_name):
    """
    Extract the test statistic computed based on data of {trial_index}th trial.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.

    :return:
        A scalar which is the test statistics of the neural Ising model.
    """
    test_indices_vet = one_sample_size_result_dict[trial_index]["test_indices_vet"]

    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt")
    x_y_mat = x_y_mat[test_indices_vet, :]
    x_neq_1_boolean_vet = x_y_mat[:, 0] == -1
    y_neq_1_boolean_vet = x_y_mat[:, 1] == -1

    predicted_parameter_mat = one_sample_size_result_dict[trial_index]["predicted_parameter_mat"][test_indices_vet, :]
    predicted_pmf_mat = gt.pmf_collection(parameter_mat=predicted_parameter_mat).numpy()

    # calculate and extract pmf of x|z [p(x = 1 | z), p(x = -1 | z)]
    pmf_x_z_mat = np.vstack([predicted_pmf_mat[:, :2].sum(axis=1), predicted_pmf_mat[:, 2:].sum(axis=1)]).T
    pmf_x_z_vet = pmf_x_z_mat[:, 0]
    pmf_x_z_vet[x_neq_1_boolean_vet] = pmf_x_z_mat[x_neq_1_boolean_vet, 1]

    # Extract relevant p(x, y | z)

    index_vet = np.zeros(len(test_indices_vet), dtype=int)
    index_vet[(~x_neq_1_boolean_vet) & y_neq_1_boolean_vet] = 1
    index_vet[x_neq_1_boolean_vet & (~y_neq_1_boolean_vet)] = 2
    index_vet[x_neq_1_boolean_vet & y_neq_1_boolean_vet] = 3

    pmf_y_x_z_vet = predicted_pmf_mat[np.arange(len(test_indices_vet)), index_vet]

    test_statistics = (np.log(pmf_y_x_z_vet) - np.log(pmf_x_z_vet)).sum()

    return test_statistics


# import pickle
# sample_size = 50
# with open("results/result_dict/mixture_data/mixture_data_16_40_breg_alt_test_prop:0.05_result_dict.p", 'rb')  as f:
#     one_sample_size_result_dict = pickle.load(f)[sample_size]
#
# trial_index = 2
# scenario = "alt"
# data_directory_name = "mixture_data"
# test = ising_powerful_test_statistic_one_trial(trial_index=trial_index,
#                                                one_sample_size_result_dict=one_sample_size_result_dict,
#                                                sample_size=sample_size, scenario=scenario,
#                                                data_directory_name=data_directory_name)

def i_projection_equations(x, predicted_parameter_list):
    return [predicted_parameter_list[0] - x[0] + np.tanh(x[1]) * predicted_parameter_list[2],
            predicted_parameter_list[1] - x[1] + np.tanh(x[0]) * predicted_parameter_list[2]]


def i_projection(predicted_parameter_mat):
    sample_size = predicted_parameter_mat.shape[0]
    projected_parameter_mat = np.zeros((sample_size, 2))
    for i in range(sample_size):
        projected_parameter_mat[i, :] = fsolve(func=i_projection_equations, x0=predicted_parameter_mat[i, :2],
                                               args=(predicted_parameter_mat[i, :]))

    return projected_parameter_mat


def ising_test_statistic_one_trial(trial_index, one_sample_size_result_dict):
    """
    Extract the test statistic computed based on data of {trial_index}th trial.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.

    :return:
        A scalar which is the test statistics of the neural Ising model.
    """

    test_indices_vet = one_sample_size_result_dict[trial_index]["test_indices_vet"]
    predicted_parameter_mat = one_sample_size_result_dict[trial_index]["predicted_parameter_mat"][test_indices_vet, :]
    projected_parameter_mat = i_projection(predicted_parameter_mat=predicted_parameter_mat)
    test_statistics = gt.kl_divergence_ising(true_parameter_mat=projected_parameter_mat,
                                             predicted_parameter_mat=predicted_parameter_mat, isAverage=True)
    # one_mat = np.array([
    #     [-1., -1., -1.],
    #     [-1, 1, 1],
    #     [1, -1, 1],
    #     [1, 1, -1]
    # ], dtype=np.float32)
    #
    # log_sum_exp_vet, reduced_log_sum_exp_vet = [], []
    # for i in test_indices_vet:
    #     parameter_vet = predicted_parameter_mat[i, :]
    #
    #     exponent_vet = tf.reduce_sum(parameter_vet * one_mat, axis=1)
    #     log_sum_exp_vet.append(tf.reduce_logsumexp(exponent_vet).numpy())
    #
    #     reduced_exponent_vet = tf.reduce_sum(parameter_vet[:2] * one_mat[:, :2], axis=1)
    #     reduced_log_sum_exp_vet.append(tf.reduce_logsumexp(reduced_exponent_vet).numpy())
    #
    # log_sum_exp_vet = np.array(log_sum_exp_vet)
    # reduced_log_sum_exp_vet = np.array(reduced_log_sum_exp_vet)
    #
    # test_statistics_vet = np.tanh(predicted_parameter_mat[test_indices_vet, 0]) * \
    #                       np.tanh(predicted_parameter_mat[test_indices_vet, 1]) * \
    #                       predicted_parameter_mat[test_indices_vet, 2] + log_sum_exp_vet - reduced_log_sum_exp_vet

    # predicted_parameter_mat = one_sample_size_result_dict[trial_index]["predicted_parameter_mat"]
    # test_indices_vet = one_sample_size_result_dict[trial_index]["test_indices_vet"]
    # test_statistics = gt.kl_divergence_ising(true_parameter_mat=predicted_parameter_mat[test_indices_vet, :2],
    #                                          predicted_parameter_mat=predicted_parameter_mat[test_indices_vet],
    #                                          isAverage=True)

    return test_statistics


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


def stratified_sq_statistic_one_trial(trial_index, one_sample_size_result_dict, isPvalue=False, cluster_number=None):
    """
    Obtain the sum of the test statistic of the stratified Chi Squared test based on data of {trial_index}th trial.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.
    :param isPvalue: A boolean. If true, return p-value. Otherwise, return the Chi-squared test statistic.
    :param cluster_number: An integer. It must be provided if isPvalue is True. Otherwise, it is ignored.

    :return:
        A scalar.
    """
    test_statistic = one_sample_size_result_dict[trial_index]
    if isPvalue:
        test_statistic = chi2.sf(test_statistic, df=cluster_number)

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


def bootstrap_p_value_one_trial(trial_index, one_sample_size_result_dict, train_p_value_boolean):
    """
    Obtain the P-value produced by the Bootstrap procedure.

    :param trial_index: An integer.
    :param one_sample_size_result_dict: A dictionary which contains simulation results of trials of a particular sample
        size.
    :param train_p_value_boolean: A boolean. If true, the function will extract the p-value computed on the training
        data. Otherwise, the p-value computed on the test data will be extracted.

    :return:
        A scalar between 0 and 1.
    """

    if train_p_value_boolean:
        test_statistic_key = 'train_p_value'
    else:
        test_statistic_key = 'test_p_value'

    return one_sample_size_result_dict[trial_index][test_statistic_key]


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
    if test_statistic_one_trial in {ising_powerful_test_statistic_one_trial, ising_score_test_statistic_one_trial}:
        # if test_statistic_one_trial in {ising_powerful_test_statistic_one_trial}:
        null_test_statistic_vet_one_sample_vet = pool.map(partial(test_statistic_one_trial,
                                                                  one_sample_size_result_dict=
                                                                  one_sample_size_null_result_dict,
                                                                  scenario="null",
                                                                  **kwargs),
                                                          trial_index_vet)
        alt_test_statistic_vet_one_sample_vet = pool.map(partial(test_statistic_one_trial,
                                                                 one_sample_size_result_dict=
                                                                 one_sample_size_alt_result_dict,
                                                                 scenario="alt",
                                                                 **kwargs),
                                                         trial_index_vet)
    else:
        null_test_statistic_vet_one_sample_vet = pool.map(partial(test_statistic_one_trial,
                                                                  one_sample_size_result_dict=
                                                                  one_sample_size_null_result_dict, **kwargs),
                                                          trial_index_vet)
        alt_test_statistic_vet_one_sample_vet = pool.map(partial(test_statistic_one_trial,
                                                                 one_sample_size_result_dict=
                                                                 one_sample_size_alt_result_dict, **kwargs),
                                                         trial_index_vet)

    return null_test_statistic_vet_one_sample_vet, alt_test_statistic_vet_one_sample_vet


##########################
# College Test Statistic #
##########################
def collect_test_statistic(pool, null_result_dict, alt_result_dict, test_statistic_one_trial, trial_index_vet,
                           **kwargs):
    """
    A wrapper function which calculate test statistics for all samples sizes.

    :param pool: A multiprocessing.pool.Pool instance.
    :param null_result_dict: A dictionary. It should be an output of the sf.(ising_)simulation_loop function called on
        null samples.
    :param alt_result_dict: A dictionary. It should be an output of the sf.(ising_)simulation_loop function called on
        alternative samples.
    :param test_statistic_one_sample_size_tuple: A tuple of outputs of the test_statistic_one_sample_size function.
    :param trial_index_vet: An array of integers which contains the trial indices of data used.
    :param kwargs: Additional keyword arguments to pass into the test_statistic_one_trial function if necessary.

    :return:
        test_statistic_dict: A dictionary containing lists of test statistics of all sample sizes.
    """
    test_statistic_dict = dict()
    for sample_size in null_result_dict.keys():
        one_sample_size_null_result_dict = null_result_dict[sample_size]
        one_sample_size_alt_result_dict = alt_result_dict[sample_size]

        if test_statistic_one_trial == ising_score_test_statistic_one_trial and sample_size == 50:
            trial_index_vet = np.delete(trial_index_vet, 127)

        test_statistic_one_sample_size_fun_arg_dict = {
            "pool": pool, "one_sample_size_null_result_dict": one_sample_size_null_result_dict,
            "one_sample_size_alt_result_dict": one_sample_size_alt_result_dict, "trial_index_vet": trial_index_vet,
            "test_statistic_one_trial": test_statistic_one_trial
        }
        if test_statistic_one_trial in {ising_powerful_test_statistic_one_trial, ising_score_test_statistic_one_trial}:
            test_statistic_one_sample_size_fun_arg_dict["sample_size"] = sample_size
        test_statistic_one_sample_size_tuple = \
            test_statistic_one_sample_size(**test_statistic_one_sample_size_fun_arg_dict, **kwargs)

        test_statistic_dict[sample_size] = test_statistic_one_sample_size_tuple

    return test_statistic_dict

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

        if test_statistic_one_trial == ising_score_test_statistic_one_trial and sample_size == 50:
            trial_index_vet = np.delete(trial_index_vet, 127)

        test_statistic_one_sample_size_fun_arg_dict = {
            "pool": pool, "one_sample_size_null_result_dict": one_sample_size_null_result_dict,
            "one_sample_size_alt_result_dict": one_sample_size_alt_result_dict, "trial_index_vet": trial_index_vet,
            "test_statistic_one_trial": test_statistic_one_trial
        }
        if test_statistic_one_trial in {ising_powerful_test_statistic_one_trial, ising_score_test_statistic_one_trial}:
            test_statistic_one_sample_size_fun_arg_dict["sample_size"] = sample_size
        test_statistic_one_sample_size_tuple = \
            test_statistic_one_sample_size(**test_statistic_one_sample_size_fun_arg_dict, **kwargs)

        fpr, tpr = fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, trial_index_vet)

        fpr_tpr_dict[sample_size] = [fpr, tpr]

    return fpr_tpr_dict


############
# PLot Roc #
############
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
    fig, ax = plt.subplots(1, 4, figsize=(9, 3))
    for i, sample_size in enumerate(hp.sample_size_vet):
        ax[i].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
        #    ax[0, 0].axvline(x=0.05, color="red")
        ax[i].set_title(f"Sample size {sample_size}")

    fig.suptitle(f"RoC Curves of {title}")
    fig.show()
    fig.savefig(f"./results/plots/{result_directory_name}/{title}.png")


def summary_roc_plot(fpr_tpr_dict_vet: list, method_name_vet: list, data_directory_name: str, result_plot_name: str,
                     suptitle: str, figsize):
    """
    Assuming there are only four sample size we are simulating. We plot RoC curves of all the methods in the
    method_name_vet and save the plot under the directory
    ./results/plots/{data_directory_name} with name summary_roc_{result_plot_name}.

    :param fpr_tpr_dict_vet: A python list of outputs of the fpr_tpr functions.
    :param method_name_vet: A list of strings. Each element is the name of the statistics method. The order should
        align with the one of fpr_tpr_dict_vet.
    :param data_directory_name: A string ('str' class) of the path towards the simulation data.
    :param result_plot_name: A string which is used as part of the plot file name.
    :param loc: The loc parameter of the legend method.
    :param figsize: A tuple of 2 numbers.

    :return:
        None.
    """
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    plt.subplots_adjust(right=0.85)
    for i, sample_size in enumerate(hp.sample_size_vet):
        for fpr_tpr_dict, method_name in zip(fpr_tpr_dict_vet, method_name_vet):
            ax[i].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1], label=method_name)
        ax[i].set_title(f"Sample size {sample_size}")

    fig.suptitle(suptitle)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    fig.show()
    fig.savefig(f"./results/plots/{data_directory_name}/summary_roc_{result_plot_name}.png")


###########################
# Plot Bootstrap P-values #
###########################
def bootstrap_qqplot(data_directory_name: str, scenario: str, result_dict_name: str):
    """

    :param data_directory_name:
    :param scenario:
    :param result_dict_name:
    :return:
    """
    with open(f'results/result_dict/{data_directory_name}/bootstrap_refit_reduced_{result_dict_name}_{scenario}_'
              f'result_dict.p', 'rb') as fp:
        bootstrap_result_dict = pickle.load(fp)

    train_p_value_vet = []
    test_p_value_vet = []

    for sample_size in bootstrap_result_dict.keys():
        sample_size_train_p_value_vet = []
        sample_size_test_p_value_vet = []
        for trial_index in bootstrap_result_dict[sample_size].keys():
            sample_size_train_p_value_vet.append(bootstrap_result_dict[sample_size][trial_index]["train_p_value"])
            sample_size_test_p_value_vet.append(bootstrap_result_dict[sample_size][trial_index]["test_p_value"])

        train_p_value_vet.append(sample_size_train_p_value_vet)
        test_p_value_vet.append(sample_size_test_p_value_vet)

    plt.scatter(train_p_value_vet[1], test_p_value_vet[1])
    fig_1 = sm.qqplot(data=np.array(test_p_value_vet[0]), dist=dist.uniform, line="45")
    plt.title("Train")
    fig_2 = sm.qqplot(data=np.array(test_p_value_vet[1]), dist=dist.uniform, line="45")
    plt.title("Test")

    fig_1.savefig(f"results/plots/{data_directory_name}/bootstrap_refit_reduced_{result_dict_name}_train.png")
    fig_2.savefig(f"results/plots/{data_directory_name}/bootstrap_refit_reduced_{result_dict_name}_test.png")


def bootstrap_roc_50_100(pool, data_directory_name, result_dict_name_vet, train_p_value_boolean, trial_index_vet):
    architecture_name_vet = []
    fpr_tpr_dict_vet = []

    for result_dict_name in result_dict_name_vet:
        with open(f'results/result_dict/{data_directory_name}/{result_dict_name}_null_'
                  f'result_dict.p', 'rb') as fp:
            null_result_dict = pickle.load(fp)
        with open(f'results/result_dict/{data_directory_name}/{result_dict_name}_alt_'
                  f'result_dict.p', 'rb') as fp:
            alt_result_dict = pickle.load(fp)

        fpr_tpr_dict = fpr_tpr(pool=pool, null_result_dict=alt_result_dict, alt_result_dict=null_result_dict,
                               test_statistic_one_trial=bootstrap_p_value_one_trial, trial_index_vet=trial_index_vet,
                               train_p_value_boolean=train_p_value_boolean)
        fpr_tpr_dict_vet.append(fpr_tpr_dict)
        architecture_name_vet.append(result_dict_name[:len(result_dict_name) - 7])

    fig, ax = plt.subplots(1, 2, figsize=(9, 9))
    sample_size = hp.sample_size_vet[0]
    for fpr_tpr_dict, architecture_name in zip(fpr_tpr_dict_vet, architecture_name_vet):
        ax[0].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1], label=architecture_name)
    ax[0].set_title(f"Sample size {sample_size}")

    sample_size = hp.sample_size_vet[1]
    for fpr_tpr_dict, method_name in zip(fpr_tpr_dict_vet, architecture_name_vet):
        ax[1].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1], label=method_name)
    ax[1].set_title(f"Sample size {sample_size}")

    fig.suptitle("RoC Curves")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')


def bootstrap_roc_500(pool, data_directory_name, result_dict_name_vet, train_p_value_boolean, trial_index_vet):
    architecture_name_vet = []
    fpr_tpr_dict_vet = []

    for result_dict_name in result_dict_name_vet:
        with open(f'results/result_dict/{data_directory_name}/{result_dict_name}_null_'
                  f'result_dict.p', 'rb') as fp:
            null_result_dict = pickle.load(fp)
        with open(f'results/result_dict/{data_directory_name}/{result_dict_name}_alt_'
                  f'result_dict.p', 'rb') as fp:
            alt_result_dict = pickle.load(fp)

        # Check if it makes sense to flip.
        fpr_tpr_dict = fpr_tpr(pool=pool, null_result_dict=alt_result_dict, alt_result_dict=null_result_dict,
                               test_statistic_one_trial=bootstrap_p_value_one_trial, trial_index_vet=trial_index_vet,
                               train_p_value_boolean=train_p_value_boolean)
        fpr_tpr_dict_vet.append(fpr_tpr_dict)
        architecture_name_vet.append(result_dict_name[:len(result_dict_name) - 7])

    fig, ax = plt.subplots(2)
    sample_size = 500
    for fpr_tpr_dict, architecture_name in zip(fpr_tpr_dict_vet, architecture_name_vet):
        ax.plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1], label=architecture_name)
    ax.set_title(f"Sample size {sample_size}")

    fig.suptitle("RoC Curves")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')


def power_curve(pool, data_directory_name, result_dict_name_vet, train_p_value_boolean, sample_size_int,
                trial_index_vet):
    rejected_proportion_vet = (np.arange(len(trial_index_vet)) + 1) / len(trial_index_vet)

    fig, ax = plt.subplots(2)
    for result_dict_name in result_dict_name_vet:
        with open(f'results/result_dict/{data_directory_name}/{result_dict_name}_null_'
                  f'result_dict.p', 'rb') as fp:
            null_result_dict = pickle.load(fp)
        null_result_dict = null_result_dict[sample_size_int]
        with open(f'results/result_dict/{data_directory_name}/{result_dict_name}_alt_'
                  f'result_dict.p', 'rb') as fp:
            alt_result_dict = pickle.load(fp)
        alt_result_dict = alt_result_dict[sample_size_int]

        null_p_value_vet_one_sample_vet, alt_p_value_vet_one_sample_vet = \
            test_statistic_one_sample_size(pool=pool, one_sample_size_null_result_dict=null_result_dict,
                                           one_sample_size_alt_result_dict=alt_result_dict,
                                           trial_index_vet=trial_index_vet,
                                           test_statistic_one_trial=bootstrap_p_value_one_trial,
                                           train_p_value_boolean=train_p_value_boolean)
        splitted_name_list = result_dict_name.split("_")
        architecture_name = splitted_name_list[3] + " " + splitted_name_list[4]

        null_p_value_vet_one_sample_vet = np.sort(null_p_value_vet_one_sample_vet)
        alt_p_value_vet_one_sample_vet = np.sort(alt_p_value_vet_one_sample_vet)

        ax[0].plot(null_p_value_vet_one_sample_vet, rejected_proportion_vet, label=architecture_name)
        ax[1].plot(alt_p_value_vet_one_sample_vet, rejected_proportion_vet, label=architecture_name)

    ax[0].set_title("Null")
    ax[1].set_title("Alt")
    fig.suptitle(f"Power Curves for Sample Size {sample_size_int}")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')


def test_statistic_histogram(test_statistic_list_dict, threshold, figsize, suptitle, smaller_boolean=True):
    fig, axs = plt.subplots(1, 4, figsize=figsize)
    axs = axs.ravel()
    i = 0
    for sample_size, p_value_tuple in test_statistic_list_dict.items():
        axs[i].hist(p_value_tuple[0], alpha=0.5, label="Null",
                    weights=np.ones(len(p_value_tuple[0])) / len(p_value_tuple[0]))
        if smaller_boolean:
            null_rejection_rate = sum(np.array(p_value_tuple[0]) < threshold) / len(p_value_tuple[0])
        else:
            null_rejection_rate = sum(np.array(p_value_tuple[0]) > threshold) / len(p_value_tuple[0])

        axs[i].set_ylim([0, 1])
        axs[i].hist(p_value_tuple[1], alpha=0.5, label="ALt",
                    weights=np.ones(len(p_value_tuple[0])) / len(p_value_tuple[0]))
        if smaller_boolean:
            alt_rejection_rate = sum(np.array(p_value_tuple[1]) < threshold) / len(p_value_tuple[1])
        else:
            alt_rejection_rate = sum(np.array(p_value_tuple[1]) > threshold) / len(p_value_tuple[1])

        axs[i].axvline(x=threshold, color="r")
        axs[i].set_title(f"N: {sample_size}. Null: {np.round(null_rejection_rate, 3)}. "
                         f"Alt: {np.round(alt_rejection_rate, 3)}")
        i += 1
    fig.suptitle(suptitle)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="right")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.show()
