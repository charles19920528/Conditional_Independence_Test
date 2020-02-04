import numpy as np
import tensorflow as tf
import ising_tuning_functions as it
import hyperparameters as hp
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(1)
tf.random.set_seed(1)

# trail_index_vet = np.random.randint(0, hp.number_of_trails+1, 10)
trail_index_vet = np.arange(hp.number_of_trails)
sample_size_vet = hp.sample_size_vet
hidden_1_out_dim_vet = np.array([3, 6, 6, 6])
hidden_2_out_dim_vet = np.array([3, 6, 6, 6])

epoch_mixture_alt_vet = np.array([70, 70, 25, 25])

if len(trail_index_vet) < hp.process_number:
    process_number = len(trail_index_vet)
else:
    process_number = hp.process_number

##########################################
# Fit the full model on the mixture data #
##########################################
it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt", input_dim=3,
               hidden_1_out_dim_vet=hidden_1_out_dim_vet,
               hidden_2_out_dim_vet=hidden_2_out_dim_vet, output_dim=3, epoch_vet=epoch_mixture_alt_vet,
               trail_index_vet=trail_index_vet,
               result_dict_name="mixture_full_model_6", process_number=process_number)

it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null", input_dim=3,
               hidden_1_out_dim_vet=hidden_1_out_dim_vet,
               hidden_2_out_dim_vet=hidden_2_out_dim_vet, output_dim=3, epoch_vet=epoch_mixture_alt_vet,
               trail_index_vet=trail_index_vet,
               result_dict_name="mixture_full_model_6", process_number=process_number)


mixture_full_model_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(
    path_epoch_kl_dict="tunning/mixture_full_model_result_alt_dict.p", sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet)

mixture_full_model_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(
    path_epoch_kl_dict="tunning/mixture_full_model_result_null_dict.p", sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet)


with open(f"tunning/mixture_full_model_epoch_kl_alt_dict.p", "wb") as fp:
    pickle.dump(mixture_full_model_epoch_kl_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


with open(f"tunning/mixture_full_model_epoch_kl_null_dict.p", "wb") as fp:
    pickle.dump(mixture_full_model_epoch_kl_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)




