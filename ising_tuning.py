import numpy as np
import tensorflow as tf
import ising_tuning_functions as it
import generate_train_functions as gt
import hyperparameters as hp
import os
import pickle
import multiprocessing as mp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

trial_index_vet = np.arange(120)
sample_size_vet = hp.sample_size_vet

if len(trial_index_vet) < hp.process_number:
    process_number = len(trial_index_vet)
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

epoch_vet = [500, 300, 150, 100]
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

for hidden_dim_mixture, result_dict_name in zip(hidden_dim_mixture_vet, mixture_result_dict_name_vet):
    one_layer_network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                                     "hidden_dim": hidden_dim_mixture, "output_dim": 3}
    it.tuning_wrapper(pool=pool, scenario="alt", data_directory_name="mixture_data",
                      network_model_class=gt.FullyConnectedNetwork,
                      number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=epoch_vet,
                      trial_index_vet=trial_index_vet, result_dict_name=result_dict_name,
                      network_model_class_kwargs=one_layer_network_kwargs_dict,
                      sample_size_vet=hp.sample_size_vet, learning_rate=hp.learning_rate,
                      weights_or_radius_kwargs={"cut_off_radius": hp.alt_cut_off_radius})

    print(f"hidden_dim {hidden_dim_mixture} finished.")

# Analysis results.
for mixture_result_dict_name in mixture_result_dict_name_vet:
    tuning_result_dict_name = mixture_result_dict_name + "_alt"
    it.process_plot_epoch_kl_raw_dict(pool=pool, tuning_result_dict_name=tuning_result_dict_name,
                                      sample_size_vet=sample_size_vet, trial_index_vet=trial_index_vet)

for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    it.plot_loss_kl(scenario="alt", tuning_result_dict_name=mixture_result_dict_name_vet[4],
                    trial_index_vet=[0, 10, 49, 60],
                    sample_size=sample_size, end_epoch=epoch, start_epoch=0, plot_train_loss_boolean=True,
                    plot_kl_boolean=True, plot_test_loss_boolean=True)

# When hidden dim is between 100 and 200, the kl seems to be smaller when sample size is 500 and 100. When hidden dim is
#   large, the kl is smaller when sample size is 100. Still is is pretty bad about 0.33.

# When sample size is 50 or 100, there seems to be overfitting problem.

# FUll model on the null data
# 1 layer
number_forward_layers = 1
hidden_dim_mixture_vet = [200]
mixture_result_dict_name_vet = [f"mixture_{number_forward_layers}_{hidden_dim}_{hp.learning_rate_mixture}"
                                for hidden_dim in hidden_dim_mixture_vet]

epoch_vet = [500, 300, 150, 100]
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

for hidden_dim_mixture, result_dict_name in zip(hidden_dim_mixture_vet, mixture_result_dict_name_vet):
    one_layer_network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                                     "hidden_dim": hidden_dim_mixture, "output_dim": 3}
    it.tuning_wrapper(pool=pool, scenario="null", data_directory_name="mixture_data",
                      network_model_class=gt.FullyConnectedNetwork,
                      number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=epoch_vet,
                      trial_index_vet=trial_index_vet, result_dict_name=result_dict_name,
                      network_model_class_kwargs=one_layer_network_kwargs_dict,
                      sample_size_vet=hp.sample_size_vet, learning_rate=hp.learning_rate,
                      weights_or_radius_kwargs={"cut_off_radius": hp.alt_cut_off_radius})

    print(f"hidden_dim {hidden_dim_mixture} finished.")

for mixture_result_dict_name in mixture_result_dict_name_vet:
    tuning_result_dict_name = mixture_result_dict_name + "_null"
    it.process_plot_epoch_kl_raw_dict(pool=pool, tuning_result_dict_name=tuning_result_dict_name,
                                      sample_size_vet=sample_size_vet, trial_index_vet=trial_index_vet)


##################################
# Fit null model on mixture data #
##################################
# 1 layer
number_forward_layers = 1
hidden_dim_mixture_vet = [10, 20, 50, 100, 200, 500, 700]
mixture_result_dict_name_vet = [f"mixture_reduced_model_{number_forward_layers}_{hidden_dim}_{hp.learning_rate_mixture}"
                                for hidden_dim in hidden_dim_mixture_vet]

epoch_vet = [500, 300, 150, 100]
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

for hidden_dim_mixture, result_dict_name in zip(hidden_dim_mixture_vet, mixture_result_dict_name_vet):
    one_layer_network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                                     "hidden_dim": hidden_dim_mixture, "output_dim": 2}
    it.tuning_wrapper(pool=pool, scenario="null", data_directory_name="mixture_data",
                      network_model_class=gt.FullyConnectedNetwork,
                      number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=epoch_vet,
                      trial_index_vet=trial_index_vet, result_dict_name=result_dict_name,
                      network_model_class_kwargs=one_layer_network_kwargs_dict,
                      sample_size_vet=hp.sample_size_vet, learning_rate=hp.learning_rate,
                      weights_or_radius_kwargs={"cut_off_radius": hp.alt_cut_off_radius})

    print(f"hidden_dim {hidden_dim_mixture} finished.")

# Analysis results.
for mixture_result_dict_name in mixture_result_dict_name_vet:
    tuning_result_dict_name = mixture_result_dict_name + "_null"
    it.process_plot_epoch_kl_raw_dict(pool=pool, tuning_result_dict_name=tuning_result_dict_name,
                                      sample_size_vet=sample_size_vet, trial_index_vet=trial_index_vet)
# 8.   8.  13.5 20.5
################################
# Tuning for true Ising model #
###############################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

epoch_vet = [500, 300, 150, 100]

true_result_dict_name = f"true_1_{hp.hidden_1_out_dim}_{hp.learning_rate_mixture}"
true_one_layer_network_kwargs_dict = {"number_forward_layers": 1, "input_dim": hp.dim_z,
                                      "hidden_dim": hp.hidden_1_out_dim, "output_dim": 3}
trial_index_vet = np.arange(20)
with open('data/ising_data/weights_dict.p', 'rb') as fp:
    true_weights_dict = pickle.load(fp)

it.tuning_wrapper(pool=pool, scenario="alt", data_directory_name="ising_data",
                  network_model_class=gt.FullyConnectedNetwork,
                  number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=epoch_vet,
                  trial_index_vet=trial_index_vet, result_dict_name=true_result_dict_name,
                  network_model_class_kwargs=true_one_layer_network_kwargs_dict,
                  sample_size_vet=hp.sample_size_vet, learning_rate=hp.learning_rate,
                  weights_or_radius_kwargs={"true_weights_dict": true_weights_dict})

it.process_plot_epoch_kl_raw_dict(pool=pool, tuning_result_dict_name=true_result_dict_name+"_alt",
                                      sample_size_vet=sample_size_vet, trial_index_vet=trial_index_vet)

for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    it.plot_loss_kl(scenario="alt", tuning_result_dict_name=true_result_dict_name,
                    trial_index_vet=[0, 2, 5, 19],
                    sample_size=sample_size, end_epoch=epoch, start_epoch=0, plot_train_loss_boolean=True, plot_kl_boolean=True,
                    plot_test_loss_boolean=True)

# The median optimal epochs are  1, 1, 8, 11.
# When sample size is 50 or 100, there seems to be overfitting problem.

pool.close()
pool.join()
