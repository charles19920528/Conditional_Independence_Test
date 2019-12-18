import numpy as np
import multiprocessing as mp
from functools import partial
import pickle
import hyperparameters as hp
import os
from sklearn import metrics
import matplotlib.pyplot as plt

# Only run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##############
# Draw graph #
##############
with open('./results/null_result_dict.p', 'rb') as fp:
    null_result_dict = pickle.load(fp)

with open('./results/alternative_result_dict.p', 'rb') as fp:
    alt_result_dict = pickle.load(fp)


#sample_size = 30
#null_result_sample_size_dict = null_result_dict[sample_size]
#alt_result_dict_sample_size_dict = alt_result_dict[sample_size]

def ising_test_statistic_one_sample(simulation_index, one_sample_result_dict):
    jxy_squared_vet = np.square(one_sample_result_dict[simulation_index]["ising_parameters"][:, 2])
    jxy_squared_mean = np.mean(jxy_squared_vet)

    return jxy_squared_mean


def ising_test_statistic_one_sample_size(null_dict, alt_dict, simulation_times):
    pool = mp.Pool(10)
    simulation_index_vet = range(simulation_times)

    null_test_statistic_vet_one_sample = pool.map(partial(ising_test_statistic_one_sample,
                                                          one_sample_result_dict = null_dict), simulation_index_vet)
    alt_test_statistic_vet_one_sample = pool.map(partial(ising_test_statistic_one_sample,
                                                         one_sample_result_dict = alt_dict), simulation_index_vet)

    return (null_test_statistic_vet_one_sample, alt_test_statistic_vet_one_sample)

#test_statistic_one_sample_size_tuple =  ising_test_statistic_one_sample_size(null_result_sample_size_dict,
#                                                                  alt_result_dict_sample_size_dict, hp.simulation_times)

def ising_fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, simulation_times):
    true_label = np.repeat([-1, 1], np.repeat(simulation_times, 2))
    combined_test_statistic_vet = np.concatenate(test_statistic_one_sample_size_tuple)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, combined_test_statistic_vet, pos_label=1)

    return fpr, tpr

#fpr, tpr = ising_fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, hp.simulation_times)

def ising_fpr_tpr(null_result_dict, alt_result_dict, simulation_times):
    fpr_tpr_dict = dict()
    for sample_size in null_result_dict.keys():
        null_result_sample_size_dict = null_result_dict[sample_size]
        alt_result_dict_sample_size_dict = alt_result_dict[sample_size]

        test_statistic_one_sample_size_tuple = ising_test_statistic_one_sample_size(null_result_sample_size_dict,
                                                                                    alt_result_dict_sample_size_dict,
                                                                                    simulation_times)

        fpr, tpr = ising_fpr_tpr_one_sample_size(test_statistic_one_sample_size_tuple, simulation_times)

        fpr_tpr_dict[sample_size] = [fpr, tpr]

    return fpr_tpr_dict


ising_fpr_tpr_dict = ising_fpr_tpr(null_result_dict, alt_result_dict, hp.simulation_times)

fig, ax = plt.subplots(2, 2)
sample_size = 30
ax[0, 0].plot(ising_fpr_tpr_dict[sample_size][0], ising_fpr_tpr_dict[sample_size][1])
#plt.axvline(x = 0.05)
plt.title(f"Sample size {sample_size}")


sample_size = 100
ax[0, 1].plot(ising_fpr_tpr_dict[sample_size][0], ising_fpr_tpr_dict[sample_size][1])
#plt.axvline(x = 0.05)
plt.title(f"Sample size {sample_size}")

sample_size = 500
ax[1, 0].plot(ising_fpr_tpr_dict[sample_size][0], ising_fpr_tpr_dict[sample_size][1])
#plt.axvline(x = 0.05)
plt.title(f"Sample size {sample_size}")

sample_size = 1000
ax[1, 1].plot(ising_fpr_tpr_dict[sample_size][0], ising_fpr_tpr_dict[sample_size][1])
#plt.axvline(x = 0.05)
plt.title(f"Sample size {sample_size}")

fig.show()
fig.savefig("./results/plots/ising.png")

###############################
# Graph for Naive Chi Squared #
###############################
with open('./results/naive_chisq_result_null_dict.p', 'rb') as fp:
    naive_chisq_result_alt_dict = pickle.load(fp)

with open('./results/naive_chisq_result_alt_dict.p', 'rb') as fp:
    alt_result_dict = pickle.load(fp)

def naive_chisq_test_statistic_one_sample_size(result_dict, simulation_times):
    test_statistic_vet = np.zeros(simulation_times)
    for i in range(simulation_times):
        test_statistic_vet[i] = result_dict[i][0]

    return test_statistic_vet

def naive_chisq_fpr_tpr_one_sample_size(null_result_dict, alt_result_dict, simulation_times):
    pass

def naive_chisq_test_statistic(null_result_dict, alt_result_dict, simulation_times):
    pass








"""
   for i in range(simulation_times):
        null_par_vet_squared = null_dict[i]["ising_parameters"][:, 2]**2
        alt_par_vet_squared = alt_dict[i]["ising_parameters"][:, 2]**2

        test_statistic_mat[0, i] = np.mean(null_par_vet_squared)
        test_statistic_mat[1, i] = np.mean(alt_par_vet_squared)

test = ising_test_statistic_one_sample_size(null_result_sample_size_dict, alt_result_dict_sample_size_dict, hp.simulation_times)




test_statistic_null = np.zeros( len(null_result_dict_30) )
for i in range(len(null_result_dict_30)):
    parameter_mat_squared = null_result_dict_30[i]["ising_parameters"]**2
    test_statistic_null[i] = np.mean(parameter_mat_squared[:, 2])

test_statistic_alternative = np.zeros( len(null_result_dict_30) )
for i in range(len(null_result_dict_30)):
    parameter_mat_squared = alternative_result_dict_30[i]["ising_parameters"]**2
    test_statistic_alternative[i] = np.mean(parameter_mat_squared[:, 2])

true_label = np.repeat([-1, 1], [len(test_statistic_null),len(test_statistic_alternative)])
combined_test_statistic_vet = np.concatenate((test_statistic_null, test_statistic_alternative))
fpr, tpr, thresholds = metrics.roc_curve(true_label, combined_test_statistic_vet, pos_label=1)
plt.plot(fpr, tpr)
plt.axvline(x = 0.05)
plt.show()













with open('./results/null_result_dict_100.p', 'rb') as fp:
    null_result_dict_100 = pickle.load(fp)

with open('./results/alternative_result_dict_100.p', 'rb') as fp:
    alternative_result_dict_100 = pickle.load(fp)

test_statistic_null = np.zeros( len(null_result_dict_100) )
for i in range(len(null_result_dict_100)):
    parameter_mat_squared = null_result_dict_100[i]["ising_parameters"]**2
    test_statistic_null[i] = np.mean(parameter_mat_squared[:, 2])

test_statistic_alternative = np.zeros( len(null_result_dict_100) )
for i in range(len(null_result_dict_100)):
    parameter_mat_squared = alternative_result_dict_100[i]["ising_parameters"]**2
    test_statistic_alternative[i] = np.mean(parameter_mat_squared[:, 2])

true_label = np.repeat([-1, 1], [len(test_statistic_null),len(test_statistic_alternative)])
combined_test_statistic_vet = np.concatenate((test_statistic_null, test_statistic_alternative))
fpr, tpr, thresholds = metrics.roc_curve(true_label, combined_test_statistic_vet, pos_label=1)
plt.plot(fpr, tpr)
plt.axvline(x = 0.05)




"""