import numpy as np
import tensorflow as tf
import ising_tuning_functions as it
import generate_train_fucntions as gt
import hyperparameters as hp
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(1)
tf.random.set_seed(1)

# trail_index_vet = np.random.randint(0, hp.number_of_trails+1, 10)
trail_index_vet = np.arange(hp.number_of_trails)
sample_size_vet = hp.sample_size_vet
hidden_1_out_dim = hidden_2_out_dim = hidden_3_out_dim = hidden_4_out_dim = 3
result_dict_name = f"mixture_full_model3{hidden_1_out_dim}"

epoch_mixture_vet = np.array([100, 100, 30, 30])

if len(trail_index_vet) < hp.process_number:
    process_number = len(trail_index_vet)
else:
    process_number = hp.process_number

##########################################
# Fit the full model on the mixture data #
##########################################
# 2 Layer
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt", input_dim=3,
#                hidden_1_out_dim_vet=hidden_1_out_dim_vet,
#                hidden_2_out_dim_vet=hidden_2_out_dim_vet, output_dim=3, epoch_vet=epoch_mixture_vet,
#                trail_index_vet=trail_index_vet,
#                result_dict_name="mixture_full_model", process_number=process_number)
#
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null", input_dim=3,
#                hidden_1_out_dim_vet=hidden_1_out_dim_vet,
#                hidden_2_out_dim_vet=hidden_2_out_dim_vet, output_dim=3, epoch_vet=epoch_mixture_vet,
#                trail_index_vet=trail_index_vet,
#                result_dict_name="mixture_full_model", process_number=process_number)
#
#
# mixture_full_model_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(
#     path_epoch_kl_dict="tunning/mixture_full_model_result_alt_dict.p", sample_size_vet=sample_size_vet,
#     trail_index_vet=trail_index_vet)
#
# mixture_full_model_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(
#     path_epoch_kl_dict="tunning/mixture_full_model_result_null_dict.p", sample_size_vet=sample_size_vet,
#     trail_index_vet=trail_index_vet)
#
#
# with open(f"tunning/mixture_full_model_epoch_kl_alt_dict.p", "wb") as fp:
#     pickle.dump(mixture_full_model_epoch_kl_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# with open(f"tunning/mixture_full_model_epoch_kl_null_dict.p", "wb") as fp:
#     pickle.dump(mixture_full_model_epoch_kl_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


# 3 layer
it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt",
               ising_network = gt.ThreeLayerIsingNetwork, epoch_vet=epoch_mixture_vet, trail_index_vet=trail_index_vet,
               result_dict_name=result_dict_name,
               process_number=process_number, input_dim=3, hidden_1_out_dim=hidden_1_out_dim,
               hidden_2_out_dim=hidden_2_out_dim, hidden_3_out_dim=hidden_3_out_dim, output_dim=3)

it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null",
               ising_network = gt.ThreeLayerIsingNetwork, input_dim=3, hidden_1_out_dim=hidden_1_out_dim,
               hidden_2_out_dim=hidden_2_out_dim, hidden_3_out_dim=hidden_3_out_dim, output_dim=3,
               epoch_vet=epoch_mixture_vet, trail_index_vet=trail_index_vet, result_dict_name=result_dict_name,
               process_number=process_number)

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(
    path_epoch_kl_dict=f"tunning/{result_dict_name}_result_alt_dict.p", sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet)

epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(
    path_epoch_kl_dict="tunning/mixture_full_model33_result_null_dict.p", sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet)


with open(f"tunning/{result_dict_name}_epoch_kl_alt_dict.p", "wb") as fp:
    pickle.dump(epoch_kl_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


with open(f"tunning/{result_dict_name}_epoch_kl_null_dict.p", "wb") as fp:
    pickle.dump(epoch_kl_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


# 4 Layer
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt",
#                ising_network = gt.FourLayerIsingNetwork, epoch_vet=epoch_mixture_vet, trail_index_vet=trail_index_vet,
#                result_dict_name="mixture_full_model43",
#                process_number=process_number, input_dim=3, hidden_1_out_dim=hidden_1_out_dim,
#                hidden_2_out_dim=hidden_2_out_dim, hidden_3_out_dim=hidden_3_out_dim, hidden_4_out_dim=hidden_4_out_dim,
#                output_dim=3)
#
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null",
#                ising_network = gt.FourLayerIsingNetwork,
#                epoch_vet=epoch_mixture_vet, trail_index_vet=trail_index_vet, result_dict_name="mixture_full_model43",
#                process_number=process_number, input_dim=3, hidden_1_out_dim=hidden_1_out_dim,
#                hidden_2_out_dim=hidden_2_out_dim, hidden_3_out_dim=hidden_3_out_dim, hidden_4_out_dim=hidden_4_out_dim,
#                output_dim=3,)
#
# np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
# mixture_full_model33_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(
#     path_epoch_kl_dict="tunning/mixture_full_model43_result_alt_dict.p", sample_size_vet=sample_size_vet,
#     trail_index_vet=trail_index_vet)
#
# mixture_full_model33_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(
#     path_epoch_kl_dict="tunning/mixture_full_model33_result_null_dict.p", sample_size_vet=sample_size_vet,
#     trail_index_vet=trail_index_vet)
#
#
# with open(f"tunning/mixture_full_model33_epoch_kl_alt_dict.p", "wb") as fp:
#     pickle.dump(mixture_full_model33_epoch_kl_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# with open(f"tunning/mixture_full_model33_epoch_kl_null_dict.p", "wb") as fp:
#     pickle.dump(mixture_full_model33_epoch_kl_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)




