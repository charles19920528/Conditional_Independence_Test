import pickle
import multiprocessing as mp
import matplotlib

matplotlib.use("TkAgg")

import result_analysis_functions as ra
import hyperparameters as hp

pool = mp.Pool(processes=hp.process_number)
trial_index_vet = list(range(hp.number_of_trials))

p_value_threshold = 0.05

ising_p_dict_list = []
mixture_p_dict_list = []
method_name_list = []

##################################
# Analayze the Naive Chi Squared #
##################################
method_name_list.append("Naive Chi-Sq")

# Ising data
with open('results/result_dict/ising_data/naive_chisq_null_result_dict.p', 'rb') as fp:
    naive_chisq_null_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/naive_chisq_alt_result_dict.p', 'rb') as fp:
    naive_chisq_alt_result_dict = pickle.load(fp)

naive_chisq_p_dict = ra.collect_test_statistic(pool=pool, null_result_dict=naive_chisq_null_result_dict,
                                               alt_result_dict=naive_chisq_alt_result_dict,
                                               test_statistic_one_trial=ra.naive_sq_statistic_one_trial,
                                               trial_index_vet=trial_index_vet, isPvalue=True)

ising_p_dict_list.append(naive_chisq_p_dict)

ra.test_statistic_histogram(test_statistic_list_dict=naive_chisq_p_dict, threshold=0.05, figsize=(14, 5),
                            suptitle="Naive Chi-Sq on Ising Data")
del naive_chisq_null_result_dict, naive_chisq_alt_result_dict, naive_chisq_p_dict

# Mixture data
with open('results/result_dict/mixture_data/naive_chisq_null_result_dict.p', 'rb') as fp:
    naive_chisq_null_result_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/naive_chisq_alt_result_dict.p', 'rb') as fp:
    naive_chisq_alt_result_dict = pickle.load(fp)

naive_chisq_p_dict = ra.collect_test_statistic(pool=pool, null_result_dict=naive_chisq_null_result_dict,
                                               alt_result_dict=naive_chisq_alt_result_dict,
                                               test_statistic_one_trial=ra.naive_sq_statistic_one_trial,
                                               trial_index_vet=trial_index_vet, isPvalue=True)

mixture_p_dict_list.append(naive_chisq_p_dict)

ra.test_statistic_histogram(test_statistic_list_dict=naive_chisq_p_dict, threshold=0.05, figsize=(14, 5),
                            suptitle="Naive Chi-Sq on Mixture Data")

del naive_chisq_null_result_dict, naive_chisq_alt_result_dict, naive_chisq_p_dict

######################################
# Analyze the stratified Chi Squared #
######################################
method_name_list.append("Stratified Chi-Sq")

# Ising data
with open('results/result_dict/ising_data/stratified_chisq_null_result_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/stratified_chisq_alt_result_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_p_dict = ra.collect_test_statistic(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                                    alt_result_dict=stratified_chisq_result_alt_dict,
                                                    test_statistic_one_trial=ra.stratified_sq_statistic_one_trial,
                                                    trial_index_vet=trial_index_vet, isPvalue=True,
                                                    cluster_number=hp.cluster_number)

ising_p_dict_list.append(stratified_chisq_p_dict)

ra.test_statistic_histogram(test_statistic_list_dict=stratified_chisq_p_dict, threshold=0.05, figsize=(14, 5),
                            suptitle="StratifiedChi-Sq on Ising Data")

del stratified_chisq_result_null_dict, stratified_chisq_result_alt_dict, stratified_chisq_p_dict

# Mixture data
with open('results/result_dict/mixture_data/stratified_chisq_null_result_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/stratified_chisq_alt_result_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_p_dict = ra.fpr_tpr(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                     alt_result_dict=stratified_chisq_result_alt_dict,
                                     test_statistic_one_trial=ra.stratified_sq_statistic_one_trial,
                                     trial_index_vet=trial_index_vet, isPvalue=True, cluster_number=hp.cluster_number)

mixture_p_dict_list.append(stratified_chisq_p_dict)

ra.test_statistic_histogram(test_statistic_list_dict=stratified_chisq_p_dict, threshold=0.05, figsize=(14, 5),
                            suptitle="StratifiedChi-Sq on Mixture Data")

del stratified_chisq_result_null_dict, stratified_chisq_result_alt_dict, stratified_chisq_p_dict

#########################
# Bootstrap Ising Model #
#########################
with open('results/result_dict/ising_data/reduced_model_expit_test_prop0_bootstrap.p', 'rb') as fp:
    bootstrap_ising_data_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/reduced_model_expit_test_prop0_bootstrap.p', 'rb') as fp:
    bootstrap_mixture_data_dict = pickle.load(fp)

##########
# Fisher #
##########
for key, method_name in zip(["sandwich True", "sandwich False"], ["Ising Score Sandwich", "Ising Score Fisher"]):
    method_name_list.append(method_name)
    # Ising data
    fisher_p_dict = {}
    for i, sample_size in enumerate(hp.sample_size_vet):
        fisher_p_dict[sample_size] = bootstrap_ising_data_dict[key][i]

    ising_p_dict_list.append(fisher_p_dict)

    ra.test_statistic_histogram(test_statistic_list_dict=fisher_p_dict, threshold=0.05, figsize=(14, 5),
                                suptitle=f"{method_name} on Ising Data")

    del fisher_p_dict

    # Mixture data
    fisher_p_dict = {}
    for i, sample_size in enumerate(hp.sample_size_vet):
        fisher_p_dict[sample_size] = bootstrap_mixture_data_dict[key][i]

    mixture_p_dict_list.append(fisher_p_dict)

    ra.test_statistic_histogram(test_statistic_list_dict=fisher_p_dict, threshold=0.05, figsize=(14, 5),
                                suptitle=f"{method_name} on Mixture Data")

    del fisher_p_dict

####################
# Analyze the CCIT #
####################
method_name_list.append("CCIT")

# Ising data
with open('results/result_dict/ising_data/ccit_null_result_dict.p', 'rb') as fp:
    ccit_null_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ccit_alt_result_dict.p', 'rb') as fp:
    ccit_alt_result_dict = pickle.load(fp)

ccit_p_dict = ra.collect_test_statistic(pool=pool, null_result_dict=ccit_null_result_dict,
                                        alt_result_dict=ccit_alt_result_dict,
                                        test_statistic_one_trial=ra.ccit_one_trial,
                                        trial_index_vet=trial_index_vet)

ising_p_dict_list.append(ccit_p_dict)

ra.test_statistic_histogram(test_statistic_list_dict=ccit_p_dict, threshold=0.5, figsize=(14, 5),
                            suptitle="CCIT on Ising Data", smaller_boolean=False)

del ccit_null_result_dict, ccit_alt_result_dict, ccit_p_dict

# Mixture Data
with open('results/result_dict/mixture_data/ccit_null_result_dict.p', 'rb') as fp:
    ccit_null_result_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/ccit_alt_result_dict.p', 'rb') as fp:
    ccit_alt_result_dict = pickle.load(fp)

ccit_p_dict = ra.collect_test_statistic(pool=pool, null_result_dict=ccit_null_result_dict,
                                        alt_result_dict=ccit_alt_result_dict,
                                        test_statistic_one_trial=ra.ccit_one_trial,
                                        trial_index_vet=trial_index_vet)

ising_p_dict_list.append(ccit_p_dict)

ra.test_statistic_histogram(test_statistic_list_dict=ccit_p_dict, threshold=0.5, figsize=(14, 5),
                            suptitle="CCIT on Mixture Data", smaller_boolean=False)
