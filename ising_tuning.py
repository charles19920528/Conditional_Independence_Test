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

np.random.seed(1)
tf.random.set_seed(1)

start_time = time.time()

for i, (hidden_dim_mixture, result_dict_name) in enumerate(zip(hidden_dim_mixture_vet, result_dict_name_vet)):
    it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null",
                   number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_mixture_1_vet,
                   trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
                   result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
                   cut_off_radius=hp.null_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
                   hidden_dim=hidden_dim_mixture, output_dim=3)

    it.tuning_loop(pool=pool, tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt",
                   number_of_test_samples_vet=number_of_test_samples_vet, epoch_vet=epoch_mixture_1_vet,
                   trail_index_vet=trail_index_vet, ising_network=gt.FullyConnectedNetwork,
                   result_dict_name=result_dict_name, sample_size_vet=sample_size_vet,
                   cut_off_radius=hp.alt_cut_off_radius, number_forward_elu_layers=1, input_dim=hp.dim_z,
                   hidden_dim=hidden_dim_mixture, output_dim=3)

print(f"Tunning mixture model with {number_forward_elu_layers} layers takes {time.time() - start_time} "
      f"seconds to finish.")


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
               input_dim=hp.dim_z, hidden_dim=2, output_dim=3, weights_dict=weights_dict)

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

##############
# Ising data #
##############
# True Ising model
with open(f"tunning/ising_true_result_null_dict.p", "rb") as fp:
    ising_true_result_null_dict = pickle.load(fp)
with open(f"tunning/ising_true_result_alt_dict.p", "rb") as fp:
    ising_true_result_alt_dict = pickle.load(fp)

ising_true_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=ising_true_result_null_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="ising_true_null")

ising_true_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=ising_true_result_alt_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="ising_wrong_alt")


# Misspecified Ising
with open(f"tunning/ising_wrong_result_null_dict.p", "rb") as fp:
    ising_wrong_result_null_dict = pickle.load(fp)
with open(f"tunning/ising_wrong_result_alt_dict.p", "rb") as fp:
    ising_wrong_result_alt_dict = pickle.load(fp)

ising_wrong_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=ising_wrong_result_null_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="ising_wrong_null")

ising_wrong_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=ising_wrong_result_alt_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="ising_wrong_alt")



for i, (sample_size, end_epoch) in enumerate(zip(sample_size_vet, epoch_ising_vet)):
    it.plot_loss_kl(experiment_result_dict=ising_wrong_result_null_dict, trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=10)
    it.plot_loss_kl(experiment_result_dict=ising_wrong_result_alt_dict, trail_index_vet=trail_index_to_plot_vet,
                    sample_size=sample_size, end_epoch=end_epoch, start_epoch=10)


################
# Mixture data #
################
with open(f"tunning/mixture_1_3_result_null_dict.p", "rb") as fp:
    mixture_1_3_result_null_dict = pickle.load(fp)
with open(f"tunning/mixture_1_3_result_alt_dict.p", "rb") as fp:
    mixture_1_3_result_alt_dict = pickle.load(fp)

mixture_1_3_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=mixture_1_3_result_null_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="mixture_1_3_null")

mixture_1_3_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=mixture_1_3_result_alt_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="mixture_1_3_alt")


with open(f"tunning/mixture_1_12_result_null_dict.p", "rb") as fp:
    mixture_1_12_result_null_dict = pickle.load(fp)
with open(f"tunning/mixture_1_12_result_alt_dict.p", "rb") as fp:
    mixture_1_12_result_alt_dict = pickle.load(fp)

mixture_1_12_epoch_kl_null_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=mixture_1_12_result_null_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="mixture_1_12_null")

mixture_1_12_epoch_kl_alt_dict = it.process_plot_epoch_kl_raw_dict(pool=pool,
    experiment_result_dict=mixture_1_12_result_alt_dict, sample_size_vet=sample_size_vet,
    trail_index_vet=trail_index_vet, name_epoch_kl_dict="mixture_1_12_alt")

# pool.close()
# pool.join()
