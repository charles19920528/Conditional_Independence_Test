import pickle
import result_analysis_functions as ra
import hyperparameters as hp

###########################
# Analyze the Ising model #
###########################
with open('results/result_dict/ising_data/ising_result_null_dict.p', 'rb') as fp:
    ising_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_result_alt_dict.p', 'rb') as fp:
    ising_result_alt_dict = pickle.load(fp)

ising_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = ising_result_null_dict, alt_result_dict = ising_result_alt_dict,
                             test_statistic_one_trail = ra.ising_test_statistic_one_trial)

ra.plot_roc(ising_fpr_tpr_dict, "Ising_Model", "ising_data")



# Use residual statistic
with open('results/result_dict/ising_data/ising_residual_result_null_dict.p', 'rb') as fp:
    ising_residual_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ising_residual_result_alt_dict.p', 'rb') as fp:
    ising_residual_result_alt_dict = pickle.load(fp)
ising_residual_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = ising_residual_result_null_dict,
                                         alt_result_dict = ising_residual_result_alt_dict,
                                         test_statistic_one_trail = ra.ising_residual_statistic_one_trail,
                                         number_of_trails=hp.number_of_trails,
                                         data_directory_name="ising_data")

ra.plot_roc(ising_residual_fpr_tpr_dict, "Ising_Residuals", "ising_data")

##################################
# Analayze the Naive Chi Squared #
##################################
with open('results/result_dict/ising_data/naive_chisq_result_null_dict.p', 'rb') as fp:
    naive_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/naive_chisq_result_alt_dict.p', 'rb') as fp:
    naive_chisq_result_alt_dict = pickle.load(fp)

naive_chisq_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = naive_chisq_result_null_dict,
                                   alt_result_dict = naive_chisq_result_alt_dict,
                             test_statistic_one_trail = ra.naive_sq_statistic_one_trail, isPvalue = False)

ra.plot_roc(naive_chisq_fpr_tpr_dict, "Naive_Chisq", "ising_data")


######################################
# Analyze the stratified Chi Squared #
######################################
with open('results/result_dict/ising_data/stratified_chisq_result_null_dict.p', 'rb') as fp:
    stratified_chisq_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/stratified_chisq_result_alt_dict.p', 'rb') as fp:
    stratified_chisq_result_alt_dict = pickle.load(fp)

stratified_chisq_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = stratified_chisq_result_null_dict,
                                        alt_result_dict = stratified_chisq_result_alt_dict,
                                        test_statistic_one_trail = ra.stratified_sq_statistic_one_trail)

ra.plot_roc(stratified_chisq_fpr_tpr_dict, "Stratified_Chisq", "ising_data")


####################
# Analyze the CCIT #
####################
with open('results/result_dict/ising_data/ccit_result_null_dict.p', 'rb') as fp:
    ccit_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/ccit_result_alt_dict.p', 'rb') as fp:
    ccit_result_alt_dict = pickle.load(fp)

ccit_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = ccit_result_null_dict, alt_result_dict = ccit_result_alt_dict,
                                        test_statistic_one_trail = ra.ccit_one_trail)

ra.plot_roc(ccit_fpr_tpr_dict, "CCIT", "ising_data")


########################################
# Analyze the misspecified Ising model #
########################################
with open('results/result_dict/ising_data/misspecified_ising_result_null_dict.p', 'rb') as fp:
    misspecified_ising_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/misspecified_ising_result_alt_dict.p', 'rb') as fp:
    misspecified_ising_result_alt_dict = pickle.load(fp)

misspecified_ising_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = misspecified_ising_result_null_dict,
                                             alt_result_dict = misspecified_ising_result_alt_dict,
                                             test_statistic_one_trail = ra.ising_test_statistic_one_trial)

ra.plot_roc(misspecified_ising_fpr_tpr_dict, "Misspecified_Ising_Model", "ising_data")

# Use residual statistic
with open('results/result_dict/ising_data/misspecified_ising_residual_result_null_dict.p', 'rb') as fp:
    misspecified_ising_residual_result_null_dict = pickle.load(fp)
with open('results/result_dict/ising_data/misspecified_ising_residual_result_alt_dict.p', 'rb') as fp:
    misspecified_ising_residual_result_alt_dict = pickle.load(fp)
ising_residual_fpr_tpr_dict = ra.fpr_tpr(null_result_dict = misspecified_ising_residual_result_null_dict,
                                         alt_result_dict = misspecified_ising_residual_result_alt_dict,
                                         test_statistic_one_trail = ra.ising_residual_statistic_one_trail,
                                         number_of_trails=hp.number_of_trails,
                                         data_directory_name="ising_data")

ra.plot_roc(ising_residual_fpr_tpr_dict, "Misspecified_Ising_Residuals", "ising_data")


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