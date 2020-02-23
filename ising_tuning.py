import numpy as np
import tensorflow as tf
import ising_tuning_functions as it
import generate_train_fucntions as gt
import hyperparameters as hp
import os
import pickle
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(1)
tf.random.set_seed(1)

trail_index_vet = np.arange(hp.number_of_trails)
sample_size_vet = hp.sample_size_vet
number_of_test_samples_vet = [5, 10, 50, 100]


if len(trail_index_vet) < hp.process_number:
    process_number = len(trail_index_vet)
else:
    process_number = hp.process_number

pool = mp.Pool(processes=process_number)

##########################################
# Fit the full model on the mixture data #
##########################################
# 2 layer
# epoch_mixture_vet = np.array([100, 120, 60, 60])
# number_forward_elu_layers = 2
# hidden_dim = 3
# result_dict_name = f"mixture_full_model{number_forward_elu_layers}{hidden_dim}"
#
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt", epoch_vet=epoch_mixture_vet,
#                trail_index_vet=trail_index_vet,
#                result_dict_name=result_dict_name, number_forward_elu_layers=number_forward_elu_layers,
#                process_number=process_number, input_dim=3, hidden_dim=hidden_dim, output_dim=3)
#
# np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
# epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(
#     path_epoch_kl_dict=f"tunning/{result_dict_name}_result_alt_dict.p", sample_size_vet=sample_size_vet,
#     trail_index_vet=trail_index_vet)


# 3 layer
# epoch_mixture_vet = np.array([120, 120, 60, 50])
# number_forward_elu_layers = 3
# hidden_dim = 3
# result_dict_name = f"mixture_full_model{number_forward_elu_layers}{hidden_dim}"
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt", epoch_vet=epoch_mixture_vet,
#                trail_index_vet=trail_index_vet,
#                result_dict_name=result_dict_name, number_forward_elu_layers=number_forward_elu_layers,
#                process_number=process_number, input_dim=3, hidden_dim=hidden_dim, output_dim=3)

# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null", epoch_vet=epoch_mixture_vet,
#                trail_index_vet=trail_index_vet,
#                result_dict_name=result_dict_name, number_forward_elu_layers=number_forward_elu_layers,
#                process_number=process_number, input_dim=3, hidden_dim=hidden_dim, output_dim=3)
#
# np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
# epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(
#     path_epoch_kl_dict=f"tunning/{result_dict_name}_result_alt_dict.p", sample_size_vet=sample_size_vet,
#     trail_index_vet=trail_index_vet)
# #
# epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(
#     path_epoch_kl_dict="tunning/mixture_full_model33_result_null_dict.p", sample_size_vet=sample_size_vet,
#     trail_index_vet=trail_index_vet)
#
#
# with open(f"tunning/{result_dict_name}_epoch_kl_alt_dict.p", "wb") as fp:
#     pickle.dump(epoch_kl_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# with open(f"tunning/{result_dict_name}_epoch_kl_null_dict.p", "wb") as fp:
#     pickle.dump(epoch_kl_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Plot the loss
# alt
# with open(f"tunning/mixture_full_model33_result_alt_dict.p", "rb") as fp:
#     mixture_full_model33_result_alt_dict = pickle.load(fp)
#
# trail_index_to_plot_vet = np.array([10, 505, 206, 100])
# for sample_size, epoch in zip(hp.sample_size_vet, epoch_mixture_vet):
#     it.plot_loss_kl(mixture_full_model33_result_alt_dict, trail_index_vet=trail_index_to_plot_vet, sample_size=sample_size,
#                     end_epoch=epoch, start_epoch=0)
#
# # Null
# with open(f"tunning/mixture_full_model33_result_null_dict.p", "rb") as fp:
#     mixture_full_model33_result_null_dict = pickle.load(fp)
#
# trail_index_to_plot_vet = np.array([10, 505, 206, 100])
# for sample_size, epoch in zip(hp.sample_size_vet, epoch_mixture_vet):
#     it.plot_loss_kl(mixture_full_model33_result_null_dict, trail_index_vet=trail_index_to_plot_vet, sample_size=sample_size,
#                     end_epoch=10, start_epoch=0)


#######################################
# Tuning for misspecified Ising model #
#######################################
epoch_ising_vet = np.array([200, 180, 50, 50])
with open('data/ising_data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)
it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="null",
               number_of_test_samples_vet=number_of_test_samples_vet, number_forward_elu_layers=2, input_dim=hp.dim_z,
               hidden_dim=2, output_dim=3, epoch_vet=epoch_ising_vet, trail_index_vet=trail_index_vet,
               result_dict_name="ising", sample_size_vet=sample_size_vet, weights_dict=weights_dict)

it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="alt",
               number_of_test_samples_vet=number_of_test_samples_vet, number_forward_elu_layers=2, input_dim=hp.dim_z,
               hidden_dim=2, output_dim=3, epoch_vet=epoch_ising_vet, trail_index_vet=trail_index_vet,
               result_dict_name="ising", sample_size_vet=sample_size_vet, weights_dict=weights_dict)

pool.close()
pool.join()









