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

trail_index_vet = np.arange(30)
sample_size_vet = hp.sample_size_vet

if len(trail_index_vet) < hp.process_number:
    process_number = len(trail_index_vet)
else:
    process_number = hp.process_number

pool = mp.Pool()

##########################################
# Fit the full model on the mixture data #
##########################################
# 1 layer
number_forward_elu_layers = 1
hidden_dim_mixture_vet = [30, 40]
mixture_result_dict_name_vet = [f"mixture_{number_forward_elu_layers}_{hidden_dim}_{hp.learning_rate_mixture}"
                                for hidden_dim in hidden_dim_mixture_vet]


np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

start_time = time.time()

for hidden_dim_mixture, result_dict_name in zip(hidden_dim_mixture_vet, mixture_result_dict_name_vet):
    it.tuning_loop(pool=pool, scenario="null", data_directory_name="mixture_data",
                   number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_mixture_1_vet,
                   trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
                   result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
                   cut_off_radius=hp.null_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
                   hidden_dim=hidden_dim_mixture, output_dim=3, learning_rate=hp.learning_rate_mixture)

    it.tuning_loop(pool=pool, scenario="alt", data_directory_name="mixture_data",
                   number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=hp.epoch_mixture_1_vet,
                   trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
                   result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
                   cut_off_radius=hp.alt_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
                   hidden_dim=hidden_dim_mixture, output_dim=3, learning_rate=hp.learning_rate_mixture)

print(f"Tuning mixture model with {number_forward_elu_layers} layers takes {time.time() - start_time} "
      f"seconds to finish.")

it.process_plot_epoch_kl_raw_dict(pool, scenario="null", result_dict_name="mixture_1_40_0.01",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
it.process_plot_epoch_kl_raw_dict(pool, scenario="alt", result_dict_name="mixture_1_16_0.01",
                                  sample_size_vet=sample_size_vet, trail_index_vet=trail_index_vet)
