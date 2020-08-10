import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import probplot
import scipy.stats.distributions as dist

import result_analysis_functions as ra
import hyperparameters as hp

pool = mp.Pool(processes=hp.process_number)
trial_index_vet = list(range(hp.number_of_trials))

##################################
# Analayze the Naive Chi Squared #
##################################
# Ising data
with open('results/result_dict/ising_data/naive_chisq_null_result_dict.p', 'rb') as fp:
    naive_chisq_null_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/naive_chisq_alt_result_dict.p', 'rb') as fp:
    naive_chisq_alt_result_dict = pickle.load(fp)

naive_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=naive_chisq_null_result_dict,
                                      alt_result_dict=naive_chisq_alt_result_dict,
                                      test_statistic_one_trial=ra.naive_sq_statistic_one_trial,
                                      trial_index_vet=trial_index_vet, isPvalue=False)

ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "ising_data")

# Mixture data
with open('results/result_dict/mixture_data/naive_chisq_null_result_dict.p', 'rb') as fp:
    naive_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/naive_chisq_alt_result_dict.p', 'rb') as fp:
    naive_chisq_result_alt_dict = pickle.load(fp)

naive_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=naive_chisq_result_null_dict,
                                      alt_result_dict=naive_chisq_result_alt_dict,
                                      test_statistic_one_trial=ra.naive_sq_statistic_one_trial,
                                      trial_index_vet=trial_index_vet, isPvalue=False)

ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "mixture_data")


######################################
# Analyze the stratified Chi Squared #
######################################
# Ising data
with open('results/result_dict/ising_data/stratified_chisq_null_result_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/stratified_chisq_alt_result_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                           alt_result_dict=stratified_chisq_result_alt_dict,
                                           test_statistic_one_trial=ra.stratified_sq_statistic_one_trial,
                                           trial_index_vet=trial_index_vet)

ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "ising_data")

# Mixture data
with open('results/result_dict/mixture_data/stratified_chisq_null_result_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/stratified_chisq_alt_result_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                           alt_result_dict=stratified_chisq_result_alt_dict,
                                           test_statistic_one_trial=ra.stratified_sq_statistic_one_trial,
                                           trial_index_vet=trial_index_vet)

ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "mixture_data")


####################################################
# Analyze the Ising model fitted on the Ising data #
####################################################
with open('results/result_dict/ising_data/ising_data_true_architecture_null_result_dict.p', 'rb') as fp:
    ising_true_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_data_true_architecture_alt_result_dict.p', 'rb') as fp:
    ising_true_result_alt_dict = pickle.load(fp)

ising_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_true_result_null_dict,
                                alt_result_dict=ising_true_result_alt_dict,
                                test_statistic_one_trial=ra.ising_test_statistic_one_trial,
                                trial_index_vet=trial_index_vet)

ra.plot_roc(ising_fpr_tpr_dict, f"True_Ising_Model", "ising_data")


######################################################
# Analyze the Ising model fitted on the mixture data #
######################################################
result_dict_name = f"mixture_data_{hp.mixture_number_forward_layer}_{hp.mixture_hidden_dim}"

with open(f'results/result_dict/mixture_data/{result_dict_name }_null_result_dict.p', 'rb') as fp:
    ising_mixture_null_result_dict = pickle.load(fp)
with open(f'results/result_dict/mixture_data/{result_dict_name }_alt_result_dict.p', 'rb') as fp:
    ising_mixture_alt_result_dict = pickle.load(fp)

ising_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_mixture_null_result_dict,
                                alt_result_dict=ising_mixture_alt_result_dict,
                                test_statistic_one_trial=ra.ising_test_statistic_one_trial,
                                trial_index_vet=trial_index_vet)

ra.plot_roc(ising_fpr_tpr_dict, result_dict_name, "mixture_data")


####################
# Analyze the CCIT #
####################
# Ising data
with open('results/result_dict/ising_data/ccit_null_result_dict.p', 'rb') as fp:
    ccit_null_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ccit_alt_result_dict.p', 'rb') as fp:
    ccit_alt_result_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ccit_null_result_dict, alt_result_dict=ccit_alt_result_dict,
                               test_statistic_one_trial=ra.ccit_one_trial, trial_index_vet=trial_index_vet)

ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "ising_data")

# Mixture data
with open('results/result_dict/mixture_data/ccit_null_result_dict.p', 'rb') as fp:
    ccit_null_result_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/ccit_alt_result_dict.p', 'rb') as fp:
    ccit_alt_result_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ccit_null_result_dict, alt_result_dict=ccit_alt_result_dict,
                               test_statistic_one_trial=ra.ccit_one_trial,trial_index_vet=trial_index_vet)

ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "mixture_data")



#############
# Bootstrap #
#############
with open('results/result_dict/ising_data/bootstrap_refit_reduced_true_architecture_50_100_null_result_dict.p', 'rb') \
        as fp:
    bootstrap_refit_true_50_100_null_dict = pickle.load(fp)

null_train_p_value_vet = []
null_test_p_value_vet = []
for sample_size in bootstrap_refit_true_50_100_null_dict.keys():
    sample_size_train_p_value_vet = []
    sample_size_test_p_value_vet = []
    for trial_index in bootstrap_refit_true_50_100_null_dict[sample_size].keys():
        sample_size_train_p_value_vet.append(bootstrap_refit_true_50_100_null_dict[sample_size][trial_index]
                                            ["train_p_value"])
        sample_size_test_p_value_vet.append(bootstrap_refit_true_50_100_null_dict[sample_size][trial_index]
                                            ["test_p_value"])

    null_train_p_value_vet.append(sample_size_train_p_value_vet)
    null_test_p_value_vet.append(sample_size_test_p_value_vet)

plt.scatter(null_train_p_value_vet[1], null_test_p_value_vet[1])

probplot(null_train_p_value_vet[0], (0, 1), dist.uniform, plot=plt)
probplot(null_train_p_value_vet[1], (0, 1), dist.uniform, plot=plt)