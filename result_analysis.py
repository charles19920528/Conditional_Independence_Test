import pickle
import multiprocessing as mp
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

mixture_fpr_tpr_dict_vet.append(naive_chisq_fpr_tpr_dict)

ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "mixture_data")


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

mixture_fpr_tpr_dict_vet.append(stratified_chisq_fpr_tpr_dict)

ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "mixture_data")


####################################################
# Analyze the Ising model fitted on the Ising data #
####################################################
method_name_vet.append("Ising")

with open('results/result_dict/ising_data/ising_data_true_architecture_null_result_dict.p', 'rb') as fp:
    ising_true_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_data_true_architecture_alt_result_dict.p', 'rb') as fp:
    ising_true_result_alt_dict = pickle.load(fp)

ising_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_true_result_null_dict,
                                alt_result_dict=ising_true_result_alt_dict,
                                test_statistic_one_trial=ra.ising_test_statistic_one_trial,
                                trial_index_vet=trial_index_vet)

ising_fpr_tpr_dict_vet.append(ising_fpr_tpr_dict)

ra.plot_roc(ising_fpr_tpr_dict, f"True_Ising_Model", "ising_data")


######################################################
# Analyze the Ising model fitted on the mixture data #
######################################################
result_dict_name = f"mixture_data_{hp.mixture_number_forward_layer}_{hp.mixture_hidden_dim}"

with open(f'results/result_dict/mixture_data/{result_dict_name }_null_result_dict.p', 'rb') as fp:
    ising_mixture_null_result_dict = pickle.load(fp)
with open(f'results/result_dict/mixture_data/{result_dict_name }_alt_result_dict.p', 'rb') as fp:
    ising_mixture_alt_result_dict = pickle.load(fp)

ising_mixture_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ising_mixture_null_result_dict,
                                alt_result_dict=ising_mixture_alt_result_dict,
                                test_statistic_one_trial=ra.ising_test_statistic_one_trial,
                                trial_index_vet=trial_index_vet)

mixture_fpr_tpr_dict_vet.append(ising_mixture_fpr_tpr_dict)

ra.plot_roc(ising_mixture_fpr_tpr_dict, result_dict_name, "mixture_data")


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

ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "ising_data")

# Mixture data
with open('results/result_dict/mixture_data/ccit_null_result_dict.p', 'rb') as fp:
    ccit_null_result_dict = pickle.load(fp)
with open('results/result_dict/mixture_data/ccit_alt_result_dict.p', 'rb') as fp:
    ccit_alt_result_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(pool=pool, null_result_dict=ccit_null_result_dict, alt_result_dict=ccit_alt_result_dict,
                               test_statistic_one_trial=ra.ccit_one_trial,trial_index_vet=trial_index_vet)

mixture_fpr_tpr_dict_vet.append(ccit_fpr_tpr_dict)

ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "mixture_data")

ra.summary_roc_plot(fpr_tpr_dict_vet=ising_fpr_tpr_dict_vet, method_name_vet=method_name_vet,
                 data_directory_name="ising_data", result_plot_name="ising")
ra.summary_roc_plot(fpr_tpr_dict_vet=mixture_fpr_tpr_dict_vet, method_name_vet=method_name_vet,
                 data_directory_name="mixture_data", result_plot_name="mixture")

#############
# Bootstrap #
#############
trial_index_vet = list(range(200))
ra.bootstrap_qqplot(data_directory_name="ising_data", scenario="null", result_dict_name="nfl:10_hd:40_50_100")

nfl_hd_vet = [(1, 10), (1, 100), (1, 200), (2, 40), (10, 40)]
result_dict_name_vet = [f"bootstrap_refit_reduced_nfl:{number_forward_layers}_hd:{hidden_dim}_50_100" for
                        number_forward_layers, hidden_dim in nfl_hd_vet]

ra.bootstrap_roc_50_100(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet,
                        train_p_value_boolean=True, trial_index_vet=trial_index_vet)

ra.power_curve(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet,
               sample_size_int=100, train_p_value_boolean=False, trial_index_vet=trial_index_vet)

nfl_hd_vet = [(1, 10), (1, 100), (1, 200), (10, 40)]
result_dict_name_vet = [f"bootstrap_refit_reduced_nfl:{number_forward_layers}_hd:{hidden_dim}_500" for
                        number_forward_layers, hidden_dim in nfl_hd_vet]
ra.bootstrap_roc_500(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet[0:2],
                        train_p_value_boolean=True, trial_index_vet=trial_index_vet)

ra.power_curve(pool=pool, data_directory_name="ising_data", result_dict_name_vet=result_dict_name_vet,
               sample_size_int=500, train_p_value_boolean=False, trial_index_vet=trial_index_vet)

pool.close()
pool.join()