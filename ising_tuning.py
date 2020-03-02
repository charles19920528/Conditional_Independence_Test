import numpy as np
import tensorflow as tf
import ising_tuning_functions as it
import generate_train_fucntions as gt
import hyperparameters as hp
import os
import pickle
import multiprocessing as mp
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

trail_index_vet = np.arange(hp.number_of_trails)
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
number_forward_elu_layers = 1
hidden_dim_mixture_vet = [12, 16]
mixture_result_dict_name_vet = [f"mixture_{number_forward_elu_layers}_{hidden_dim}_{hp.learning_rate_mixture}"
                                for hidden_dim in hidden_dim_mixture_vet]

np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

start_time = time.time()

for hidden_dim_mixture, result_dict_name in zip(hidden_dim_mixture_vet, mixture_result_dict_name_vet):
    # it.tuning_loop(pool=pool, scenario="null", data_directory_name="mixture_data",
    #                number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_mixture_1_vet,
    #                trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
    #                result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
    #                cut_off_radius=hp.null_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
    #                hidden_dim=hidden_dim_mixture, output_dim=3, learning_rate=hp.learning_rate_mixture)

    it.tuning_loop(pool=pool, scenario="alt", data_directory_name="mixture_data",
                   number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_mixture_1_vet,
                   trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
                   result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
                   cut_off_radius=hp.alt_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
                   hidden_dim=hidden_dim_mixture, output_dim=3, learning_rate=hp.learning_rate_mixture)

print(f"Tuning mixture model with {number_forward_elu_layers} layers takes {time.time() - start_time} "
      f"seconds to finish.")


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
               result_dict_name=f"ising_true_rate_{hp.learning_rate}", sample_size_vet=sample_size_vet, weights_dict=weights_dict,
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
               number_of_test_samples_vet=hp.number_of_test_samples_vet,  epoch_vet=hp.epoch_ising_vet,
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
trail_index_to_plot_vet = [0,1,2,3]

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
