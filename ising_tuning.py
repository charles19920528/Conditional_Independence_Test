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
number_of_test_samples_vet = [5, 10, 50, 100]

epoch_ising_vet = np.array([300, 250, 100, 100])


if len(trail_index_vet) < hp.process_number:
    process_number = len(trail_index_vet)
else:
    process_number = hp.process_number

pool = mp.Pool(processes=process_number)

##########################################
# Fit the full model on the mixture data #
##########################################
# 1 layer
epoch_mixture_1_vet = np.array([100, 120, 60, 60])
number_forward_elu_layers = 1
hidden_dim_mixture_vet = [3, 6, 12]
result_dict_name_vet = [f"mixture_{number_forward_elu_layers}_{hidden_dim}" for hidden_dim in hidden_dim_mixture_vet]

np.random.seed(1)
tf.random.set_seed(1)

start_time = time.time()

it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null",
               number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_mixture_1_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
               result_dict_name=result_dict_name_vet[0], sample_size_vet=sample_size_vet,
               cut_off_radius=hp.null_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
               hidden_dim=hidden_dim_mixture_vet[0], output_dim=3)

it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt",
               number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_mixture_1_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
               result_dict_name=result_dict_name_vet[0], sample_size_vet=sample_size_vet,
               cut_off_radius=hp.alt_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
               hidden_dim=hidden_dim_mixture_vet[0], output_dim=3)

print("Tunning true Ising model takes %s seconds to finish." % (time.time() - start_time))

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

################################
# Tuning for true Ising model #
###############################
np.random.seed(1)
tf.random.set_seed(1)

with open('data/ising_data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)

start_time = time.time()

it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="null",
               number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.IsingNetwork,
               result_dict_name="ising_true", sample_size_vet=sample_size_vet, weights_dict=weights_dict,
               input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="alt",
               number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.IsingNetwork,
               result_dict_name="ising_true", sample_size_vet=sample_size_vet, weights_dict=weights_dict,
               input_dim=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim, output_dim=3)

print("Tunning true Ising model takes %s seconds to finish." % (time.time() - start_time))

#######################################
# Tuning for misspecified Ising model #
#######################################
np.random.seed(1)
tf.random.set_seed(1)

with open('data/ising_data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)

start_time = time.time()

it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="null",
               number_of_test_samples_vet=number_of_test_samples_vet,  epoch_vet=epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
               result_dict_name="ising_wrong", sample_size_vet=sample_size_vet, number_forward_elu_layers=2,
               input_dim=hp.dim_z, hidden_dim=2, output_dim=3,weights_dict=weights_dict)

it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="alt",
               number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_ising_vet,
               trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
               result_dict_name="ising_wrong", sample_size_vet=sample_size_vet, number_forward_elu_layers=2,
               input_dim=hp.dim_z, hidden_dim=2, output_dim=3, weights_dict=weights_dict)

print("Tunning misspecified Ising model takes %s seconds to finish." % (time.time() - start_time))

###################
# Result analysis #
###################
trail_index_to_plot_vet = [0,290,360,402]
# Misspecified Ising
ising_wrong_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    path_epoch_kl_dict=f"tunning/ising_wrong_result_null_dict.p", sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet)

ising_wrong_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    path_epoch_kl_dict=f"tunning/ising_wrong_result_alt_dict.p", sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet)

with open(f"tunning/ising_wrong_result_null_dict.p", "rb") as fp:
    ising_wrong_result_null_dict = pickle.load(fp)
with open(f"tunning/ising_wrong_result_alt_dict.p", "rb") as fp:
    ising_wrong_result_alt_dict = pickle.load(fp)

for i, (sample_size, end_epoch) in enumerate(zip(sample_size_vet, epoch_ising_vet)):
    it.plot_loss_kl(experiment_result_dict=ising_wrong_result_null_dict, trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=10)
    it.plot_loss_kl(experiment_result_dict=ising_wrong_result_alt_dict, trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=10)


# pool.close()
# pool.join()
