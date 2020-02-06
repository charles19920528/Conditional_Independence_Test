import numpy as np
import generate_train_fucntions as gt
import ising_tuning_functions as it
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
import hyperparameters as hp


# Test code
# trail_index = 3
# scenario = "alt"
# sample_size = 1000
# epoch = 20

# with open('data/ising_data/weights_dict.p', 'rb') as fp:
#     weights_dict = pickle.load(fp)
# it.tuning_pool_wrapper_ising_data(trail_index=trail_index, scenario=scenario, sample_size=sample_size, epoch=epoch,
#                              weights_dict=weights_dict, input_dim=3, hidden_1_out_dim=2, hidden_2_out_dim=2,
#                              output_dim=3)

# it.tuning_pool_wrapper_mixture_data(trail_index=trail_index, scenario=scenario, sample_size=sample_size, epoch=epoch,
#                                     input_dim=3, hidden_1_out_dim=2, hidden_2_out_dim=2, output_dim=3)


# epoch_vet_ising = np.int32(hp.epoch_vet_misspecified * 1.3)
trail_index_vet = np.array([100, 105, 106, 200])

# Ising data_tuning
# with open('data/ising_data/weights_dict.p', 'rb') as fp:
#     weights_dict = pickle.load(fp)
#
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="alt", epoch_vet=epoch_vet_ising,
#                trail_index_vet=trail_index_vet, result_dict_name="ising_data", weights_dict=weights_dict, input_dim=3,
#                hidden_1_out_dim_vet=np.repeat(2, 4), hidden_2_out_dim_vet=np.repeat(2, 4), output_dim=3)
#
# it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_ising_data, scenario="null", epoch_vet=epoch_vet_ising,
#                trail_index_vet=np.arange(4), result_dict_name="ising_data", weights_dict=weights_dict, input_dim=3,
#                hidden_1_out_dim_vet=np.repeat(2, 4), hidden_2_out_dim_vet=np.repeat(2, 4), output_dim=3)

# Mixture data tunning
epoch_vet_mixture_alt = np.array([50, 40, 20, 15])
epoch_vet_mixture_null = np.array([50, 40, 20, 15])
hidden_1_out_dim_vet = np.array([3, 3, 3, 3])
hidden_2_out_dim_vet = np.array([3, 3, 3, 3])

it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="alt",
               epoch_vet=epoch_vet_mixture_alt, trail_index_vet=trail_index_vet, result_dict_name="mixture_data",
               input_dim=3, hidden_1_out_dim_vet=hidden_1_out_dim_vet, hidden_2_out_dim_vet=hidden_2_out_dim_vet,
               output_dim=3, process_number=4)

it.tuning_loop(tunning_pool_wrapper=it.tuning_pool_wrapper_mixture_data, scenario="null",
               epoch_vet=epoch_vet_mixture_null, trail_index_vet=trail_index_vet, result_dict_name="mixture_data",
               input_dim=3, hidden_1_out_dim_vet=hidden_1_out_dim_vet, hidden_2_out_dim_vet=hidden_2_out_dim_vet,
               output_dim=3, process_number=4)


# Fit null model
# tuning_loop(tunning_pool_wrapper=tuning_pool_wrapper_mixture, scenario="alt", epoch_vet=epoch_vet_mixture_alt,
#             trail_index_vet=trail_index_vet, result_dict_name="mixture_data_null_model", hidden_1_out_dim_vet=
#             np.array([2, 4, 6, 6]), hidden_2_out_dim_vet=np.array([2, 4, 6, 6]), output_dim=2,process_number=4)


###################
# Result analysis #
###################
# Ising data
# with open(f"tunning/ising_data_result_alt_dict.p", "rb") as fp:
#     ising_data_result_alt_dict = pickle.load(fp)
# for sample_size, epoch in zip(hp.sample_size_vet, epoch_vet_ising):
#     it.plot_loss_kl(ising_data_result_alt_dict, sample_size, end_epoch=epoch)

# Mixture data
# Alt
with open(f"tunning/mixture_data_result_alt_dict.p", "rb") as fp:
    mixture_data_result_alt_dict = pickle.load(fp)

for sample_size, epoch in zip(hp.sample_size_vet, epoch_vet_mixture_alt):
    it.plot_loss_kl(mixture_data_result_alt_dict, sample_size, end_epoch=epoch)

# Null
with open(f"tunning/mixture_data_result_null_dict.p", "rb") as fp:
    mixture_data_result_null_dict = pickle.load(fp)

for sample_size, epoch in zip(hp.sample_size_vet, epoch_vet_mixture_null):
    it.plot_loss_kl(mixture_data_result_null_dict, sample_size, end_epoch=epoch)
