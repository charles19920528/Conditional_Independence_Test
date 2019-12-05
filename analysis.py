import numpy as np
import pickle
import os
from sklearn import metrics
import matplotlib.pyplot as plt

# Only run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


with open('./results/null_result_dict_30.p', 'rb') as fp:
    null_result_dict_30 = pickle.load(fp)

with open('./results/alternative_result_dict_30.p', 'rb') as fp:
    alternative_result_dict_30 = pickle.load(fp)

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



