import pickle
import multiprocessing as mp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import result_analysis_functions as ra
import hyperparameters as hp

import time

pool = mp.Pool(processes=hp.process_number - 12)
trial_index_vet = list(range(hp.number_of_trials))

##############
# Ising Data #
##############
for test_prop in hp.test_prop_list:
    with open(f'results/result_dict/ising_data/ising_data_reduced_model_true_architecture_null_test_prop:{test_prop}_result_dict.p',
              'rb') as fp:
        ising_true_result_null_dict = pickle.load(fp)
    with open(f'results/result_dict/ising_data/ising_data_reduced_model_true_architecture_alt_test_prop:{test_prop}_result_dict.p',
              'rb') as fp:
        ising_true_result_alt_dict = pickle.load(fp)

    ising_p_value_dict = {}
    sandwich_boolean_vet = [True, False]
    for sandwich_boolean in sandwich_boolean_vet:
        p_value_tuple_list = []
        for sample_size in hp.sample_size_vet:
            time1 = time.time()
            one_sample_size_p_value_tuple = ra.test_statistic_one_sample_size(
                pool=pool, one_sample_size_null_result_dict=ising_true_result_null_dict[sample_size],
                one_sample_size_alt_result_dict=ising_true_result_alt_dict[sample_size],
                test_statistic_one_trial=ra.ising_score_test_statistic_one_trial,
                trial_index_vet=trial_index_vet, data_directory_name="ising_data",
                sample_size=sample_size, sandwich_boolean=sandwich_boolean, n_batches=6, batch_size=90)
            time2 = time.time()
            p_value_tuple_list.append(one_sample_size_p_value_tuple)
            print(f"Ising Data. sandwich {sandwich_boolean}, sample size {sample_size} finished.")
            print(f"time {time2 - time1}")
        ising_p_value_dict[f"sandwich {sandwich_boolean}"] = p_value_tuple_list


    with open(f"./results/result_dict/ising_data/reduced_model_expit_test_prop{test_prop}_bootstrap.p", "wb") as fp:
        pickle.dump(ising_p_value_dict, fp, protocol=4)

# del p_value_tuple_list
#

for test_prop in hp.test_prop_list:
    with open(f'results/result_dict/ising_data/reduced_model_expit_test_prop{test_prop}_bootstrap.p', 'rb') as fp:
        ising_p_value_dict = pickle.load(fp)
    for sandwich_boolean in [True, False]:
        p_value_tuple_list = ising_p_value_dict[f"sandwich {sandwich_boolean}"]

        fig, axs = plt.subplots(1, 4, figsize=(15, 6))
        axs = axs.ravel()
        for i, (p_value_tuple, sample_size) in enumerate(zip(p_value_tuple_list, hp.sample_size_vet)):
            axs[i].hist(p_value_tuple[0], alpha=0.5, label="Null", weights=np.ones(len(p_value_tuple[0])) / len(p_value_tuple[0]))
            axs[i].hist(p_value_tuple[1], alpha=0.5, label="ALt", weights=np.ones(len(p_value_tuple[0])) / len(p_value_tuple[0]))
            axs[i].legend()
            axs[i].set_title(f"Sample Size {sample_size}")
        fig.suptitle(f"Ising Data, Sandwich: {sandwich_boolean}, Test Prop: {test_prop}")
        fig.show()

################
# Mixture data #
################
for test_prop in hp.test_prop_list:
    with open(f'results/result_dict/mixture_data/mixture_data_reduced_model_{hp.reduced_model_mixture_number_forward_layer_null}_'
              f'{hp.reduced_model_mixture_hidden_dim_null}_null_test_prop:{test_prop}_result_dict.p', 'rb') as fp:
        null_ising_mixture_result_dict = pickle.load(fp)
    with open(f'results/result_dict/mixture_data/mixture_data_reduced_model_{hp.reduced_model_mixture_number_forward_layer_alt}_'
              f'{hp.reduced_model_mixture_hidden_dim_alt}_alt_test_prop:{test_prop}_result_dict.p', 'rb') as fp:
        alt_ising_mixture_result_dict = pickle.load(fp)

    mixture_p_value_dict = {}
    sandwich_boolean_vet = [True, False]
    for sandwich_boolean in sandwich_boolean_vet:
        p_value_tuple_list = []
        for sample_size in hp.sample_size_vet:
            time1 = time.time()
            one_sample_size_p_value_tuple = ra.test_statistic_one_sample_size(
                pool=pool, one_sample_size_null_result_dict=null_ising_mixture_result_dict[sample_size],
                one_sample_size_alt_result_dict=alt_ising_mixture_result_dict[sample_size],
                test_statistic_one_trial=ra.ising_score_test_statistic_one_trial,
                trial_index_vet=trial_index_vet, data_directory_name="mixture_data",
                sample_size=sample_size, sandwich_boolean=sandwich_boolean, n_batches=9, batch_size=60)
            time2 = time.time()
            p_value_tuple_list.append(one_sample_size_p_value_tuple)
            print(f"Mixture data, sandwich {sandwich_boolean}, sample size {sample_size}, test:{test_prop} finished.")
            print(f"time {time2-time1}")
        mixture_p_value_dict[f"sandwich {sandwich_boolean}"] = p_value_tuple_list

    with open(f"./results/result_dict/mixture_data/reduced_model_expit_test_prop{test_prop}_bootstrap.p", "wb") as fp:
        pickle.dump(mixture_p_value_dict, fp, protocol=4)


for test_prop in hp.test_prop_list:
    with open(f'results/result_dict/mixture_data/reduced_model_expit_test_prop{test_prop}_bootstrap.p', 'rb') as fp:
        mixture_p_value_dict = pickle.load(fp)
    for sandwich_boolean in [True, False]:
        p_value_tuple_list = mixture_p_value_dict[f"sandwich {sandwich_boolean}"]

        fig, axs = plt.subplots(1, 4, figsize=(15, 6))
        axs = axs.ravel()
        for i, (p_value_tuple, sample_size) in enumerate(zip(p_value_tuple_list, hp.sample_size_vet)):
            axs[i].hist(p_value_tuple[0], alpha=0.5, label="Null",
                        weights=np.ones(len(p_value_tuple[0])) / len(p_value_tuple[0]))
            axs[i].hist(p_value_tuple[1], alpha=0.5, label="ALt",
                        weights=np.ones(len(p_value_tuple[0])) / len(p_value_tuple[0]))
            axs[i].legend()
            axs[i].set_title(f"Sample Size {sample_size}")
        fig.suptitle(f"Mixture Data, Sandwich: {sandwich_boolean}, Test Prop: {test_prop}")
        fig.show()
