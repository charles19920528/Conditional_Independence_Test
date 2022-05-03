import pickle
import multiprocessing as mp
import matplotlib

matplotlib.use("TkAgg")
import result_analysis_functions as ra
import hyperparameters as hp

pool = mp.Pool(processes=hp.process_number)
trial_index_vet = list(range(hp.number_of_trials))

ising_fpr_tpr_dict_vet = []
mixture_fpr_tpr_dict_vet = []
method_name_vet = []

##################################
# Analayze the Naive Chi Squared #
##################################
method_name_vet.append("Naive Chi-Sq")

# Ising data
with open('results/result_dict/ising_data/naive_chisq_null_result_dict.p', 'rb') as fp:
    naive_chisq_null_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/naive_chisq_alt_result_dict.p', 'rb') as fp:
    naive_chisq_alt_result_dict = pickle.load(fp)

naive_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=naive_chisq_null_result_dict,
                                      alt_result_dict=naive_chisq_alt_result_dict,
                                      test_statistic_one_trial=ra.naive_sq_statistic_one_trial,
                                      trial_index_vet=trial_index_vet, isPvalue=False)

ising_fpr_tpr_dict_vet.append(naive_chisq_fpr_tpr_dict)

# ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "ising_data")

del naive_chisq_null_result_dict, naive_chisq_alt_result_dict, naive_chisq_fpr_tpr_dict

# Mixture data
with open('results/result_dict/mixture_data/naive_chisq_null_result_dict.p', 'rb') as fp:
    naive_chisq_null_result_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/naive_chisq_alt_result_dict.p', 'rb') as fp:
    naive_chisq_alt_result_dict = pickle.load(fp)

naive_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=naive_chisq_null_result_dict,
                                      alt_result_dict=naive_chisq_alt_result_dict,
                                      test_statistic_one_trial=ra.naive_sq_statistic_one_trial,
                                      trial_index_vet=trial_index_vet, isPvalue=False)

mixture_fpr_tpr_dict_vet.append(naive_chisq_fpr_tpr_dict)

# ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "mixture_data")

del naive_chisq_null_result_dict, naive_chisq_alt_result_dict, naive_chisq_fpr_tpr_dict

######################################
# Analyze the stratified Chi Squared #
######################################
method_name_vet.append("Stratified Chi-Sq")

# Ising data
with open('results/result_dict/ising_data/stratified_chisq_null_result_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/stratified_chisq_alt_result_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                           alt_result_dict=stratified_chisq_result_alt_dict,
                                           test_statistic_one_trial=ra.stratified_sq_statistic_one_trial,
                                           trial_index_vet=trial_index_vet)

ising_fpr_tpr_dict_vet.append(stratified_chisq_fpr_tpr_dict)

# ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "ising_data")

del stratified_chisq_result_null_dict, stratified_chisq_result_alt_dict, stratified_chisq_fpr_tpr_dict

# Mixture data
with open('results/result_dict/mixture_data/stratified_chisq_null_result_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/stratified_chisq_alt_result_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=stratified_chisq_result_null_dict,
                                           alt_result_dict=stratified_chisq_result_alt_dict,
                                           test_statistic_one_trial=ra.stratified_sq_statistic_one_trial,
                                           trial_index_vet=trial_index_vet)

mixture_fpr_tpr_dict_vet.append(stratified_chisq_fpr_tpr_dict)

# ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "mixture_data")

del stratified_chisq_result_null_dict, stratified_chisq_result_alt_dict, stratified_chisq_fpr_tpr_dict

############
# Ising Kl #
############
method_name_vet.append("Ising KL")

# Ising Data
with open('results/result_dict/ising_data/ising_data_full_model_true_architecture_alt_test_prop:0.1_result_dict.p',
          'rb') as fp:
    ising_true_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_data_full_model_true_architecture_alt_test_prop:0.1_result_dict.p',
          'rb') as fp:
    ising_true_result_alt_dict = pickle.load(fp)

ising_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_true_result_null_dict,
                                alt_result_dict=ising_true_result_alt_dict,
                                test_statistic_one_trial=ra.ising_test_statistic_one_trial,
                                trial_index_vet=trial_index_vet)

ising_fpr_tpr_dict_vet.append(ising_fpr_tpr_dict)

# ra.plot_roc(ising_fpr_tpr_dict, f"Ising True Architecture Breg", "ising_data")

del ising_fpr_tpr_dict

######################################################
# Analyze the Ising model fitted on the mixture data #
######################################################
# plot_title = "ising_model_mixture_data_breg"

with open(f'results/result_dict/mixture_data/mixture_data_full_model_{hp.full_model_mixture_number_forward_layer_null}_'
          f'{hp.full_model_mixture_hidden_dim_null}_null_test_prop:0.1_result_dict.p', 'rb') as fp:
    null_ising_mixture_result_dict = pickle.load(fp)
with open(f'results/result_dict/mixture_data/mixture_data_full_model_{hp.full_model_mixture_number_forward_layer_alt}_'
          f'{hp.full_model_mixture_hidden_dim_alt}_alt_test_prop:0.1_result_dict.p', 'rb') as fp:
    alt_ising_mixture_result_dict = pickle.load(fp)

ising_mixture_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=null_ising_mixture_result_dict,
                                        alt_result_dict=alt_ising_mixture_result_dict,
                                        test_statistic_one_trial=ra.ising_test_statistic_one_trial,
                                        trial_index_vet=trial_index_vet)

mixture_fpr_tpr_dict_vet.append(ising_mixture_fpr_tpr_dict)

# ra.plot_roc(ising_mixture_fpr_tpr_dict, plot_title, "mixture_data")

del ising_mixture_fpr_tpr_dict

######################################
# Ising Model with MP test statistic #
######################################
method_name_vet.append("Ising MP")

# Ising Data
ising_mp_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_true_result_null_dict,
                                   alt_result_dict=ising_true_result_alt_dict,
                                   test_statistic_one_trial=ra.ising_powerful_test_statistic_one_trial,
                                   trial_index_vet=trial_index_vet, data_directory_name="ising_data")

ising_fpr_tpr_dict_vet.append(ising_mp_fpr_tpr_dict)
# ra.plot_roc(ising_mp_fpr_tpr_dict, f"Ising True Architecture MP", "ising_data")

del ising_mp_fpr_tpr_dict

# Mixture Data
ising_mixture_mp_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=null_ising_mixture_result_dict,
                                           alt_result_dict=alt_ising_mixture_result_dict,
                                           test_statistic_one_trial=ra.ising_powerful_test_statistic_one_trial,
                                           trial_index_vet=trial_index_vet, data_directory_name="mixture_data")

mixture_fpr_tpr_dict_vet.append(ising_mixture_mp_fpr_tpr_dict)

del ising_mixture_mp_fpr_tpr_dict
del null_ising_mixture_result_dict, ising_true_result_null_dict, alt_ising_mixture_result_dict, \
    ising_true_result_alt_dict

###################################################
# Ising Model with score test statistic last layer #
###################################################
############
# Sandwich #
############
method_name_vet.append("Ising Score Sandwich")

# Ising Data
# with open('results/result_dict/ising_data/ising_data_true_architecture_null_test_prop:0_result_dict.p',
#           'rb') as fp:
#     null_ising_true_score_result_dict = pickle.load(fp)
# with open('results/result_dict/ising_data/ising_data_true_architecture_alt_test_prop:0_result_dict.p',
#           'rb') as fp:
#     alt_ising_true_score_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_data_reduced_model_true_architecture_null_test_prop:0_result_dict.p',
          'rb') as fp:
    null_ising_reduced_model_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_data_reduced_model_true_architecture_alt_test_prop:0_result_dict.p',
          'rb') as fp:
    alt_ising_reduced_model_result_dict = pickle.load(fp)
ising_score_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=null_ising_reduced_model_result_dict,
                                      alt_result_dict=alt_ising_reduced_model_result_dict,
                                      test_statistic_one_trial=ra.ising_score_test_statistic_one_trial,
                                      trial_index_vet=trial_index_vet, data_directory_name="ising_data",
                                      sandwich_boolean=True)

ising_fpr_tpr_dict_vet.append(ising_score_fpr_tpr_dict)
# ra.plot_roc(ising_score_fpr_tpr_dict, f"Ising Reduced Model True Architecture Score", "ising_data")

# Mixture Data
# with open(f'results/result_dict/mixture_data/mixture_data_{hp.mixture_number_forward_layer_null}_'
#           f'{hp.mixture_hidden_dim_null}_null_test_prop:0_result_dict.p', 'rb') as fp:
#     null_ising_mixture_score_result_dict = pickle.load(fp)
# with open(f'results/result_dict/mixture_data/mixture_data_{hp.mixture_number_forward_layer_alt}_'
#           f'{hp.mixture_hidden_dim_alt}_alt_test_prop:0_result_dict.p', 'rb') as fp:
#     alt_ising_mixture_score_result_dict = pickle.load(fp)

with open(f'results/result_dict/mixture_data/'
          f'mixture_data_reduced_model_{hp.reduced_model_mixture_number_forward_layer_null}_'
          f'{hp.reduced_model_mixture_hidden_dim_null}_null_test_prop:0_result_dict.p', 'rb') as fp:
    null_mixture_reduced_model_result_dict = pickle.load(fp)
with open(
        f'results/result_dict/mixture_data/mixture_data_reduced_model_{hp.reduced_model_mixture_number_forward_layer_alt}_'
        f'{hp.reduced_model_mixture_hidden_dim_alt}_alt_test_prop:0_result_dict.p', 'rb') as fp:
    alt_mixture_reduced_model_result_dict = pickle.load(fp)

ising_mixture_score_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=null_mixture_reduced_model_result_dict,
                                              alt_result_dict=alt_mixture_reduced_model_result_dict,
                                              test_statistic_one_trial=ra.ising_score_test_statistic_one_trial,
                                              trial_index_vet=trial_index_vet, data_directory_name="mixture_data",
                                              sandwich_boolean=True)

mixture_fpr_tpr_dict_vet.append(ising_mixture_score_fpr_tpr_dict)
# ra.plot_roc(ising_mixture_score_fpr_tpr_dict, f"Ising Reduced Model Score", "mixture_data")

del ising_score_fpr_tpr_dict, ising_mixture_score_fpr_tpr_dict

###########
# Fischer #
###########
method_name_vet.append("Ising Score Fisher")
# Ising Data
ising_score_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=null_ising_reduced_model_result_dict,
                                      alt_result_dict=alt_ising_reduced_model_result_dict,
                                      test_statistic_one_trial=ra.ising_score_test_statistic_one_trial,
                                      trial_index_vet=trial_index_vet, data_directory_name="ising_data",
                                      sandwich_boolean=False)

ising_fpr_tpr_dict_vet.append(ising_score_fpr_tpr_dict)

# Mixture Data
ising_mixture_score_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=null_mixture_reduced_model_result_dict,
                                              alt_result_dict=alt_mixture_reduced_model_result_dict,
                                              test_statistic_one_trial=ra.ising_score_test_statistic_one_trial,
                                              trial_index_vet=trial_index_vet, data_directory_name="mixture_data",
                                              sandwich_boolean=False)

mixture_fpr_tpr_dict_vet.append(ising_mixture_score_fpr_tpr_dict)

del ising_score_fpr_tpr_dict, ising_mixture_score_fpr_tpr_dict
del null_ising_reduced_model_result_dict, alt_ising_reduced_model_result_dict, null_mixture_reduced_model_result_dict, \
    alt_mixture_reduced_model_result_dict

###########################################################
# Analyze Different Train and Test Split for Ising Models #
###########################################################
# mixture_ising_model_fpr_tpr_dict_list = []
# for test_prop in hp.test_prop_list:
#     # with open(f'results/result_dict/ising_data/ising_data_true_architecture_null_test_prop:{test_prop}_result_dict.p',
#     #           'rb') as fp:
#     #     null_ising_mixture_result_dict = pickle.load(fp)
#     # with open(f'results/result_dict/ising_data/ising_data_true_architecture_alt_test_prop:{test_prop}_result_dict.p',
#     #           'rb') as fp:
#     #     alt_ising_mixture_result_dict = pickle.load(fp)
#
#     with open(f'results/result_dict/mixture_data/mixture_data_{hp.mixture_number_forward_layer_null}_'
#               f'{hp.mixture_hidden_dim_null}_breg_null_test_prop:{test_prop}_result_dict.p', 'rb') as fp:
#         null_ising_mixture_result_dict = pickle.load(fp)
#     with open(f'results/result_dict/mixture_data/mixture_data_{hp.mixture_number_forward_layer_alt}_'
#               f'{hp.mixture_hidden_dim_alt}_breg_alt_test_prop:{test_prop}_result_dict.p', 'rb') as fp:
#         alt_ising_mixture_result_dict = pickle.load(fp)
#
#     ising_mixture_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=null_ising_mixture_result_dict,
#                                             alt_result_dict=alt_ising_mixture_result_dict,
#                                             test_statistic_one_trial=ra.ising_test_statistic_one_trial,
#                                             trial_index_vet=trial_index_vet)
#     mixture_ising_model_fpr_tpr_dict_list.append(ising_mixture_fpr_tpr_dict)
#
# ra.summary_roc_plot(fpr_tpr_dict_vet=mixture_ising_model_fpr_tpr_dict_list, method_name_vet=hp.test_prop_list,
#                     data_directory_name="mixture_data", result_plot_name="ising_model_with_different_test_prop_under "
#                                                                          "mixture_data_(breg)")

####################
# Analyze the CCIT #
####################
method_name_vet.append("CCIT")

# Ising data
with open('results/result_dict/ising_data/ccit_null_result_dict.p', 'rb') as fp:
    ccit_null_result_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ccit_alt_result_dict.p', 'rb') as fp:
    ccit_alt_result_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ccit_null_result_dict, alt_result_dict=ccit_alt_result_dict,
                               test_statistic_one_trial=ra.ccit_one_trial, trial_index_vet=trial_index_vet)

ising_fpr_tpr_dict_vet.append(ccit_fpr_tpr_dict)

# ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "ising_data")
del ccit_null_result_dict, ccit_alt_result_dict, ccit_fpr_tpr_dict

# Mixture data
with open('results/result_dict/mixture_data/ccit_null_result_dict.p', 'rb') as fp:
    ccit_null_result_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/ccit_alt_result_dict.p', 'rb') as fp:
    ccit_alt_result_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ccit_null_result_dict, alt_result_dict=ccit_alt_result_dict,
                               test_statistic_one_trial=ra.ccit_one_trial, trial_index_vet=trial_index_vet)

mixture_fpr_tpr_dict_vet.append(ccit_fpr_tpr_dict)

# ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "mixture_data")
del ccit_null_result_dict, ccit_alt_result_dict, ccit_fpr_tpr_dict

######################
# Save and Summarize #
######################
with open(f"./results/result_dict/ising_data/ising_fpr_tpr_vet.p", "wb") as fp:
    pickle.dump(ising_fpr_tpr_dict_vet, fp, protocol=4)
with open(f"./results/result_dict/mixture_data/mixture_fpr_tpr_vet.p", "wb") as fp:
    pickle.dump(mixture_fpr_tpr_dict_vet, fp, protocol=4)

ra.summary_roc_plot(fpr_tpr_dict_vet=ising_fpr_tpr_dict_vet, method_name_vet=method_name_vet,
                    data_directory_name="ising_data", result_plot_name="ising_data_summary",
                    suptitle="RoC Curves under Ising Data", figsize=(15, 4))
ra.summary_roc_plot(fpr_tpr_dict_vet=mixture_fpr_tpr_dict_vet, method_name_vet=method_name_vet,
                    data_directory_name="mixture_data", result_plot_name="mixture_data_summary",
                    suptitle="RoC Curves under Mixture Data", figsize=(15, 4))

#############
# Bootstrap #
#############
# trial_index_vet = list(range(200))
# ra.bootstrap_qqplot(data_directory_name="ising_data", scenario="null", result_dict_name="nfl:10_hd:40_50_100")
#
# nfl_hd_vet = [(1, 10), (1, 100), (1, 200), (2, 40), (10, 40)]
# result_dict_name_vet = [f"bootstrap_refit_reduced_nfl:{number_forward_layers}_hd:{hidden_dim}_50_100" for
#                         number_forward_layers, hidden_dim in nfl_hd_vet]
#
# ra.bootstrap_roc_50_100(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet,
#                         train_p_value_boolean=True, trial_index_vet=trial_index_vet)
#
# ra.power_curve(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet,
#                sample_size_int=100, train_p_value_boolean=False, trial_index_vet=trial_index_vet)
#
# nfl_hd_vet = [(1, 10), (1, 100), (1, 200), (10, 40), (10, 100)]
# result_dict_name_vet = [f"bootstrap_refit_reduced_nfl:{number_forward_layers}_hd:{hidden_dim}_500" for
#                         number_forward_layers, hidden_dim in nfl_hd_vet]
# ra.bootstrap_roc_500(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet[0:2],
#                      train_p_value_boolean=True, trial_index_vet=trial_index_vet)
# ra.power_curve(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet,
#                sample_size_int=500, train_p_value_boolean=False, trial_index_vet=trial_index_vet)

pool.close()
pool.join()
