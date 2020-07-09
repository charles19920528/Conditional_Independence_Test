import numpy as np
import tensorflow as tf
import ising_tuning_functions as it
import generate_train_functions_nightly as gt
import hyperparameters as hp
import os
import pickle
import multiprocessing as mp
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

trail_index_vet = np.arange(120)
sample_size_vet = hp.sample_size_vet

if len(trail_index_vet) < hp.process_number:
    process_number = len(trail_index_vet)
else:
    process_number = hp.process_number

pool = mp.Pool(processes=process_number)

##########################################
# Fit the full model on the mixture data #
##########################################
# 1 layer
number_forward_layers = 1
hidden_dim_mixture_vet = [10, 20, 50, 100, 200, 500, 700]
mixture_result_dict_name_vet = [f"mixture_{number_forward_layers}_{hidden_dim}_{hp.learning_rate_mixture}"
                                for hidden_dim in hidden_dim_mixture_vet]

max_epoch_vet = [500, 300, 150, 100]
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

for hidden_dim_mixture, result_dict_name in zip(hidden_dim_mixture_vet, mixture_result_dict_name_vet):
    one_layer_network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                                     "hidden_dim": hidden_dim_mixture, "output_dim": 3}
    it.tuning_wrapper(pool=pool, scenario="alt", data_directory_name="mixture_data",
                      network_model_class=gt.FullyConnectedNetwork,
                      number_of_test_samples_vet=hp.number_of_test_samples_vet, max_epoch_vet=max_epoch_vet,
                      trial_index_vet=trail_index_vet, result_dict_name=result_dict_name,
                      network_model_class_kwargs=one_layer_network_kwargs_dict,
                      sample_size_vet=hp.sample_size_vet, learning_rate=hp.learning_rate,
                      weights_or_radius_kwargs={"cut_off_radius": hp.alt_cut_off_radius})

    print(f"hidden_dim {hidden_dim_mixture} finished.")


# Analysis results.
for mixture_result_dict_name in mixture_result_dict_name_vet:
    raw_result_dict_name = mixture_result_dict_name + "_alt"
    it.process_plot_epoch_kl_raw_dict(pool=pool, raw_result_dict_name=raw_result_dict_name,
                                      sample_size_vet=sample_size_vet, trial_index_vet=trail_index_vet)

it.plot_loss_kl(scenario="alt", raw_result_dict_name=mixture_result_dict_name_vet[4], trial_index_vet=[0,10, 49, 60],
                sample_size=1000, end_epoch=50, start_epoch=0,
                 plot_train_loss=True, plot_kl=True, plot_test_loss=True)

# When hidden dim is between 100 and 200, the kl seems to be smaller when sample size is 500 and 100. When hidden dim is
#   large, the kl is smaller when sample size is 100. Still is is pretty bad about 0.33.






















# Code below needs to be replaced.

################################
# Tuning for true Ising model #
###############################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

with open('data/ising_data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)

start_time = time.time()

it.tuning_loop(pool=pool, scenario="null", data_directory_name="ising_data",
               number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.IsingNetwork,
               result_dict_name=f"ising_true_rate_{hp.learning_rate}", sample_size_vet=sample_size_vet,
               weights_dict=weights_dict, input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

it.tuning_loop(pool=pool, scenario="alt", data_directory_name="ising_data",
               number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.IsingNetwork,
               result_dict_name=f"ising_true_rate_{hp.learning_rate}", sample_size_vet=sample_size_vet,
               weights_dict=weights_dict,
               input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

print("Tuning true Ising model takes %s seconds to finish." % (time.time() - start_time))

# Tuning for the test data size when sample size is 100
for number_of_test_samples in hp.number_of_test_samples_100_vet:
    it.tuning_loop(pool=pool, scenario="alt", data_directory_name="ising_data",
                   number_of_test_samples_vet=[number_of_test_samples], epoch_vet=[400],
                   trail_index_vet=trail_index_vet, ising_network=gt.IsingNetwork,
                   result_dict_name=f"ising_true_rate_{hp.learning_rate}_n_100_test_{number_of_test_samples}",
                   sample_size_vet=[100], weights_dict=weights_dict, input_dim=hp.dim_z,
                   hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

    it.tuning_loop(pool=pool, scenario="null", data_directory_name="ising_data",
                   number_of_test_samples_vet=[number_of_test_samples], epoch_vet=[400],
                   trail_index_vet=trail_index_vet, ising_network=gt.IsingNetwork,
                   result_dict_name=f"ising_true_rate_{hp.learning_rate}_n_100_test_{number_of_test_samples}",
                   sample_size_vet=[100], weights_dict=weights_dict, input_dim=hp.dim_z,
                   hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

#######################################
# Tuning for misspecified Ising model #
#######################################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

with open('data/ising_data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)

start_time = time.time()

it.tuning_loop(pool=pool, scenario="null", data_directory_name="ising_data",
               number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
               result_dict_name="ising_wrong", sample_size_vet=sample_size_vet, number_forward_elu_layers=2,
               input_dim=hp.dim_z, hidden_dim=2, output_dim=3, weights_dict=weights_dict)

it.tuning_loop(pool=pool, scenario="alt", data_directory_name="ising_data",
               number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
               result_dict_name="ising_wrong", sample_size_vet=sample_size_vet, number_forward_elu_layers=2,
               input_dim=hp.dim_z, hidden_dim=2, output_dim=3, weights_dict=weights_dict)

print("Tuning misspecified Ising model takes %s seconds to finish." % (time.time() - start_time))

###################
# Result analysis #
###################
trail_index_to_plot_vet = [0, 1, 2, 3]

##############
# Ising data #
##############
# True Ising model
it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name=f"ising_true_rate_{hp.learning_rate}",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name=f"ising_true_rate_{hp.learning_rate}",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)

# Analyze test sample size when the total sample size is 100.
result_dict_name_test_size_vet = [f"ising_true_rate_{hp.learning_rate}_n_100_test_{number_of_test_samples}"
                                  for number_of_test_samples in hp.number_of_test_samples_100_vet]

for result_dict_name in result_dict_name_test_size_vet:
    it.process_plot_epoch_kl_raw_dict(pool=pool, scenario="alt", result_dict_name=result_dict_name,
                                      sample_size_vet=[100], trail_index_vet=trail_index_vet)

for result_dict_name in result_dict_name_test_size_vet:
    it.process_plot_epoch_kl_raw_dict(pool=pool, scenario="null", result_dict_name=result_dict_name,
                                      sample_size_vet=[100], trail_index_vet=trail_index_vet)

# Misspecified Ising
it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name=f"ising_wrong_rate_{hp.learning_rate}",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name=f"ising_wrong_rate_{hp.learning_rate}",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)

################
# Mixture data #
################
it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name="mixture_1_12_0.01",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name="mixture_1_12_0.01",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)

it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name="mixture_1_16_0.01",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name="mixture_1_16_0.01",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)

for result_dict_name in mixture_result_dict_name_vet:
    it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name=result_dict_name,
                                      sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
    # it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name=result_dict_name,
    #                                   sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)

for i, (sample_size, end_epoch) in enumerate(zip(sample_size_vet, hp.epoch_mixture_1_vet)):
    it.plot_loss_kl(scenario="null", result_dict_name="mixture_1_12_0.01", trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=0)
    it.plot_loss_kl(scenario="alt", result_dict_name="mixture_1_12_0.01", trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=0)

# for result_dict_name in mixture_result_dict_name_vet:
#     it.plot_loss_kl(scenario="null", result_dict_name=result_dict_name, trail_index_vet=trail_index_to_plot_vet,
#                     sample_size=1000, end_epoch=100, start_epoch=0)
#     it.plot_loss_kl(scenario="alt", result_dict_name=result_dict_name, trail_index_vet=trail_index_to_plot_vet,
#                     sample_size=1000, end_epoch=100, start_epoch=0)

pool.close()
pool.join()
