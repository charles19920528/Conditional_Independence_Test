import numpy as np
import multiprocessing as mp
from functools import partial
import pickle
import os
# Only run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn import metrics
import matplotlib.pyplot as plt
import hyperparameters as hp



################################
# Analysis the signal strength #
################################
"""
alt_network = dg.alt_network_generate

kl_list = []
for i, sample_size in enumerate(hp.sample_size_vet):
    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}.txt", dtype = np.float32)
    alt_parameter_mat = alt_network(z_mat)
    kl_list.append( gt.kl_divergence(true_parameter_mat = alt_parameter_mat,
                                  predicted_parameter_mat = alt_parameter_mat[:, :2], isAverage = False) )

fig0, ax0 = plt.subplots()
ax0.set_title("Boxplot of KL-Divergence")
ax0.boxplot(kl_list)
ax0.set_xticklabels(hp.sample_size_vet)
fig0.savefig("./results/plots/kl_boxplot.png")

"""

############################
# Analysis the Ising model #
############################
with open('./results/ising_result_null_dict.p', 'rb') as fp:
    ising_result_null_dict = pickle.load(fp)

with open('./results/ising_result_alt_dict.p', 'rb') as fp:
    ising_result_alt_dict = pickle.load(fp)


# sample_size = 30
# null_result_sample_size_dict = ising_result_null_dict[sample_size]
# alt_result_dict_sample_size_dict = ising_result_alt_dict[sample_size]

def ising_test_statistic_one_sample(simulation_index, one_sample_size_result_dict):
    jxy_squared_vet = np.square(one_sample_size_result_dict[simulation_index][:, 2])
    jxy_squared_mean = np.mean(jxy_squared_vet)

    return jxy_squared_mean

# ising_result_alt_dict

def test_statistic_one_sample_size(null_dict, alt_dict, simulation_times, test_statistic_one_sample,**kwargs):
    pool = mp.Pool(hp.process_number)
    simulation_index_vet = range(simulation_times)

    null_test_statistic_vet_one_sample = pool.map(partial(test_statistic_one_sample,
                                                          one_sample_size_result_dict = null_dict, **kwargs),
                                                  simulation_index_vet)
    alt_test_statistic_vet_one_sample = pool.map(partial(test_statistic_one_sample,
                                                         one_sample_size_result_dict = alt_dict, **kwargs),
                                                 simulation_index_vet)

    return (null_test_statistic_vet_one_sample, alt_test_statistic_vet_one_sample)

# test_statistic_one_sample_size_tuple =  ising_test_statistic_one_sample_size(null_result_sample_size_dict,
#                                                                   alt_result_dict_sample_size_dict, hp.simulation_times)

def fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, simulation_times):
    true_label = np.repeat([-1, 1], np.repeat(simulation_times, 2))
    combined_test_statistic_vet = np.concatenate(test_statistic_one_sample_size_tuple)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, combined_test_statistic_vet, pos_label=1)

    return fpr, tpr

# fpr, tpr = fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, hp.simulation_times)

def fpr_tpr(null_result_dict, alt_result_dict, simulation_times, test_statistic_one_sample,**kwargs):
    fpr_tpr_dict = dict()
    for sample_size in null_result_dict.keys():
        null_result_sample_size_dict = null_result_dict[sample_size]
        alt_result_dict_sample_size_dict = alt_result_dict[sample_size]

        test_statistic_one_sample_size_tuple = test_statistic_one_sample_size(null_dict = null_result_sample_size_dict,
                                                                         alt_dict = alt_result_dict_sample_size_dict,
                                                                         simulation_times = simulation_times,
                                                                         test_statistic_one_sample =
                                                                         test_statistic_one_sample, **kwargs)

        fpr, tpr = fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, simulation_times)

        fpr_tpr_dict[sample_size] = [fpr, tpr]

    return fpr_tpr_dict


ising_fpr_tpr_dict = fpr_tpr(null_result_dict = ising_result_null_dict, alt_result_dict = ising_result_alt_dict,
                             simulation_times = hp.simulation_times,
                             test_statistic_one_sample = ising_test_statistic_one_sample)

def plot_roc(fpr_tpr_dict, model_for_main_title):
    fig, ax = plt.subplots(2, 2)
    sample_size = 30
    ax[0, 0].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    # plt.axvline(x = 0.05)
    ax[0, 0].set_title(f"Sample size {sample_size}")

    sample_size = 100
    ax[0, 1].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    # plt.axvline(x = 0.05)
    ax[0, 1].set_title(f"Sample size {sample_size}")

    sample_size = 500
    ax[1, 0].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    # plt.axvline(x = 0.05)
    ax[1, 0].set_title(f"Sample size {sample_size}")

    sample_size = 1000
    ax[1, 1].plot(fpr_tpr_dict[sample_size][0], fpr_tpr_dict[sample_size][1])
    # plt.axvline(x = 0.05)
    ax[1, 1].set_title(f"Sample size {sample_size}")

    fig.suptitle(model_for_main_title)
    fig.show()
    fig.savefig(f"./results/plots/{model_for_main_title}.png")

plot_roc(ising_fpr_tpr_dict, "Ising_Model")

###################################
# Analaysis the Naive Chi Squared #
###################################
with open('./results/naive_chisq_result_null_dict.p', 'rb') as fp:
    naive_chisq_result_null_dict = pickle.load(fp)

with open('./results/naive_chisq_result_alt_dict.p', 'rb') as fp:
    naive_chisq_result_alt_dict = pickle.load(fp)

def naive_sq_statistic_one_sample(simulation_index, one_sample_size_result_dict, isPvalue):
    if isPvalue:
        pvalue = one_sample_size_result_dict[simulation_index][1]
        return pvalue
    else:
        chisquare_statistic = one_sample_size_result_dict[simulation_index][0]
        return chisquare_statistic


    return jxy_squared_mean


naive_chisq_fpr_tpr_dict = fpr_tpr(null_result_dict = naive_chisq_result_null_dict,
                                   alt_result_dict = naive_chisq_result_alt_dict,
                             simulation_times = hp.simulation_times,
                             test_statistic_one_sample = naive_sq_statistic_one_sample, isPvalue = False)

plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq")

# The behavior when sample size is 500 is surprisingly good. The test statistic is larger under the alternative.


########################################
# Analaysis the stratified Chi Squared #
########################################
with open('./results/stratified_chisq_result_null_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)

with open('./results/stratified_chisq_result_alt_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

def stratified_sq_statistic_one_sample(simulation_index, one_sample_size_result_dict):
    pvalue = one_sample_size_result_dict[simulation_index]
    return pvalue


stratified_chisq_fpr_tpr_dict = fpr_tpr(null_result_dict = stratified_chisq_result_null_dict,
                                        alt_result_dict = stratified_chisq_result_alt_dict,
                                        simulation_times = hp.simulation_times,
                                        test_statistic_one_sample = stratified_sq_statistic_one_sample)
plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq")




# See if test statistics are the same.
naiv_null, _ = test_statistic_one_sample_size(naive_chisq_result_null_dict[30], naive_chisq_result_alt_dict[30],
                                              hp.simulation_times,
               naive_sq_statistic_one_sample, isPvalue = False)

stra_null, _ = test_statistic_one_sample_size(stratified_chisq_result_null_dict[30], stratified_chisq_result_alt_dict[30],
                                              hp.simulation_times, stratified_sq_statistic_one_sample)


test = [x in stra_null for x in naiv_null]