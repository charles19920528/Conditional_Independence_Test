import numpy as np
from scipy.stats import chi2_contingency
import hyperparameters as hp
import pandas as pd
from functools import partial
import multiprocessing as mp
import time


def chi_squared_test(x_y_mat):
    """
    Perform unconditional Chi-squared test.
    :param x_y_mat: A 2-d numpy array with columns corresponding to x and y.
    :return: result_vet: A length 2 tuple. result_vet[0] is the test statistic and result_vet[1] is the p-value.
    """
    x_vet = x_y_mat[:, 0]
    y_vet = x_y_mat[:, 1]
    contingency_table = pd.crosstab(x_vet, y_vet, rownames="x", colnames="y")

    result_vet = chi2_contingency(contingency_table)[0:2]
    return result_vet


"""
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
x_y_mat = np.loadtxt(f"./data/null/x_y_mat_1000_0.txt")
t = np.loadtxt(f"./data/null/x_y_mat_1000_2.txt")
####################
# Parallel version #
####################
def simulation_wrap(simulation, scenario, sample_size):
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation}.txt")
    result_vet = chi_squared_test(x_y_mat)
    return (simulation, result_vet)

t = partial(simulation_wrap, sample_size=30, scenario="null")
t(100)
start_time = time.time()
naive_chisq_result_null_dict = dict()
for sample_size in hp.sample_size_vet:

    pool = mp.Pool(10)
    simulation_vet = range(hp.simulation_times)
    pool_result_vet = pool.map(partial(simulation_wrap, sample_size=sample_size, scenario="null"), simulation_vet)

    sample_result_dict = dict(pool_result_vet)
    naive_chisq_result_null_dict[sample_size] = sample_result_dict
print("--- %s seconds ---" % (time.time() - start_time))


naive_chisq_result_null_dict = dict()
for sample_size in hp.sample_size_vet:

    sample_result_dict = dict()
    for simulation in range(hp.simulation_times):
        x_y_mat = np.loadtxt(f"./data/null/x_y_mat_{sample_size}_{simulation}.txt")
        sample_result_dict[simulation] = chi_squared_test(x_y_mat)

    naive_chisq_result_null_dict[sample_size] = sample_result_dict


pool = mp.Pool(processes=10)
results = []

[((0 ,0), 0), ((0, 1), 1)]
test=dict([((0 ,0), 0), ((0, 1), 1)])


[x for x in test.keys() if x[1] == 1]


names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']


