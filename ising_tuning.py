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
number_of_test_samples_vet = [10, 10, 50, 100]

epoch_ising_vet = np.array([300, 300, 120, 120])
epoch_mixture_1_vet = np.array([300, 250, 100, 100])

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
hidden_dim_mixture_vet = [3, 12]
result_dict_name_vet = [f"mixture_{number_forward_elu_layers}_{hidden_dim}" for hidden_dim in hidden_dim_mixture_vet]

np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

start_time = time.time()

for i, (hidden_dim_mixture, result_dict_name) in enumerate(zip(hidden_dim_mixture_vet, result_dict_name_vet)):
    it.tuning_loop(pool=pool, scenario="null",
                   number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_mixture_1_vet,
                   trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
                   result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
                   cut_off_radius=hp.null_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
                   hidden_dim=hidden_dim_mixture, output_dim=3)

    it.tuning_loop(pool=pool, scenario="alt",
                   number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_mixture_1_vet,
                   trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
                   result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
                   cut_off_radius=hp.alt_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
                   hidden_dim=hidden_dim_mixture, output_dim=3)

print(f"Tunning mixture model with {number_forward_elu_layers} layers takes {time.time() - start_time} "
      f"seconds to finish.")

###################
# Result analysis #
###################
trail_index_to_plot_vet = [0,290,360,402]

##############
# Ising data #
##############
# True Ising model
it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name=f"ising_true_rate_{hp.learning_rate}",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name=f"ising_true_rate_{hp.learning_rate}",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)


# Misspecified Ising
it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name="ising_wrong",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name="ising_wrong",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)


################
# Mixture data #
################
it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name="mixture_1_3",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name="mixture_1_3",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)

it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name="mixture_1_12",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name="mixture_1_12",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)

for i, (sample_size, end_epoch) in enumerate(zip(sample_size_vet, epoch_ising_vet)):
    it.plot_loss_kl(scenario="null", result_dict_name="mixture_1_12", trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=10)
    it.plot_loss_kl(scenario="alt", result_dict_name="mixture_1_12", trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=10)


pool.close()
pool.join()
