import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt

import result_analysis_functions as ra
import hyperparameters as hp

pool = mp.Pool(processes=hp.process_number)

##################################
# Analayze the Naive Chi Squared #
##################################
# Ising data
with open('results/result_dict/ising_data/naive_chisq_result_null_dict.p', 'rb') as fp:
    naive_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/naive_chisq_result_alt_dict.p', 'rb') as fp:
    naive_chisq_result_alt_dict = pickle.load(fp)

naive_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=naive_chisq_result_null_dict,
                                      alt_result_dict=naive_chisq_result_alt_dict,
                                      test_statistic_one_trail=ra.naive_sq_statistic_one_trail, isPvalue=False)

ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "ising_data")

# Mixture data
with open('results/result_dict/mixture_data/naive_chisq_result_null_dict.p', 'rb') as fp:
    naive_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/naive_chisq_result_alt_dict.p', 'rb') as fp:
    naive_chisq_result_alt_dict = pickle.load(fp)

naive_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=naive_chisq_result_null_dict,
                                      alt_result_dict=naive_chisq_result_alt_dict,
                                      test_statistic_one_trail=ra.naive_sq_statistic_one_trail, isPvalue=False)

ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "mixture_data")


######################################
# Analyze the stratified Chi Squared #
######################################
# Ising data
with open('results/result_dict/ising_data/stratified_chisq_result_null_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/stratified_chisq_result_alt_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                           alt_result_dict=stratified_chisq_result_alt_dict,
                                           test_statistic_one_trail=ra.stratified_sq_statistic_one_trail)

ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "ising_data")

# Mixture data
with open('results/result_dict/mixture_data/stratified_chisq_result_null_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/stratified_chisq_result_alt_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                           alt_result_dict=stratified_chisq_result_alt_dict,
                                           test_statistic_one_trail=ra.stratified_sq_statistic_one_trail)

ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "mixture_data")


####################################################
# Analyze the Ising model fitted on the Ising data #
####################################################
with open('results/result_dict/ising_data/ising_true_rate_0.01_result_null_dict.p', 'rb') as fp:
    ising_true_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_true_rate_0.01_result_alt_dict.p', 'rb') as fp:
    ising_true_result_alt_dict = pickle.load(fp)

ising_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_true_result_null_dict,
                                alt_result_dict=ising_true_result_alt_dict,
                                test_statistic_one_trail=ra.ising_test_statistic_one_trial)

ra.plot_roc(ising_fpr_tpr_dict, "True Ising Model, Rate: 0.01", "ising_data")

fig, ax = plt.subplots(1)
color_vet = ["cyan", "magenta", "yellow", "black"]
for i, number_of_test_samples in enumerate(hp.number_of_test_samples_100_vet):
    with open(f'results/result_dict/ising_data/ising_true_rate_{hp.learning_rate}_n_100_test_{number_of_test_samples}_'
              f'result_null_dict.p', 'rb') as fp:
        ising_true_result_null_dict = pickle.load(fp)
    with open(f'results/result_dict/ising_data/ising_true_rate_{hp.learning_rate}_n_100_test_{number_of_test_samples}_'
              f'result_alt_dict.p', 'rb') as fp:
        ising_true_result_alt_dict = pickle.load(fp)

    ising_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_true_result_null_dict,
                                    alt_result_dict=ising_true_result_alt_dict,
                                    test_statistic_one_trail=ra.ising_test_statistic_one_trial)


    sample_size = 100
    ax.plot(ising_fpr_tpr_dict[sample_size][0], ising_fpr_tpr_dict[sample_size][1], color=color_vet[i],
            label=number_of_test_samples)
    #    ax[0, 0].axvline(x=0.05, color="red")
    ax.set_title(f"Sample size 100 with different testing size")
ax.legend()


#################################################################
# Analyze the misspecified Ising model fitted on the Ising data #
#################################################################
with open('results/result_dict/ising_data/ising_wrong_rate_0.01_result_null_dict.p', 'rb') as fp:
    misspecified_ising_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_wrong_rate_0.01_result_alt_dict.p', 'rb') as fp:
    misspecified_ising_result_alt_dict = pickle.load(fp)

misspecified_ising_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=misspecified_ising_result_null_dict,
                                             alt_result_dict=misspecified_ising_result_alt_dict,
                                             test_statistic_one_trail=ra.ising_test_statistic_one_trial)

ra.plot_roc(misspecified_ising_fpr_tpr_dict, "Misspecified Ising Model, Rate: 0.01", "ising_data")


######################################################
# Analyze the Ising model fitted on the mixture data #
######################################################
# with open('results/result_dict/mixture_data/ising_wrong_rate_0.005_result_null_dict.p', 'rb') as fp:
#     ising_result_null_dict = pickle.load(fp)
# with open('results/result_dict/mixture_data/ising_wrong_rate_0.005_result_alt_dict.p', 'rb') as fp:
#     ising_result_alt_dict = pickle.load(fp)
#
# ising_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = ising_result_null_dict, alt_result_dict = ising_result_alt_dict,
#                              test_statistic_one_trail = ra.ising_test_statistic_one_trial)
#
# ra.plot_roc(ising_fpr_tpr_dict, "Ising_Model", "mixture_data")

with open('results/result_dict/mixture_data/ising_optimal_epoch_result_null_dict.p', 'rb') as fp:
    ising_optimal_epoch_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/ising_optimal_epoch_result_alt_dict.p', 'rb') as fp:
    ising_optimal_epoch_result_alt_dict = pickle.load(fp)

ising_fpr_tpr_dict = ra.fpr_tpr(null_result_dict=ising_optimal_epoch_result_null_dict,
                                alt_result_dict=ising_optimal_epoch_result_alt_dict,
                                test_statistic_one_trail=ra.ising_test_statistic_one_trial)

ra.plot_roc(ising_fpr_tpr_dict, "Ising_Model_test", "mixture_data")


####################
# Analyze the CCIT #
####################
# Ising data
with open('results/result_dict/ising_data/ccit_result_null_dict.p', 'rb') as fp:
    ccit_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ccit_result_alt_dict.p', 'rb') as fp:
    ccit_result_alt_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ccit_result_null_dict, alt_result_dict=ccit_result_alt_dict,
                               test_statistic_one_trail=ra.ccit_one_trail)

ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "ising_data")

# Mixture data
with open('results/result_dict/mixture_data/ccit_result_null_dict.p', 'rb') as fp:
    ccit_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/ccit_result_alt_dict.p', 'rb') as fp:
    ccit_result_alt_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ccit_result_null_dict, alt_result_dict=ccit_result_alt_dict,
                               test_statistic_one_trail=ra.ccit_one_trail)

ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "mixture_data")
