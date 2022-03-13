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
number_forward_layers_vet = [1, 4, 12, 16]
hidden_dim_mixture_vet = [40, 80, 160, 200, 320]
test_sample_prop = hp.test_sample_prop
epoch_vet = [500, 300, 150, 100]
mixture_result_dict_name_vet = []

np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

for scenario in ["null", "alt"]:
    for number_forward_layers in number_forward_layers_vet:
        for hidden_dim_mixture in hidden_dim_mixture_vet:
            result_dict_name = f"mixture_{number_forward_layers}_{hidden_dim_mixture}_{hp.learning_rate_mixture}_" \
                               f"test_prop:{test_sample_prop}"

            mixture_result_dict_name_vet.append(result_dict_name)

            network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                                   "hidden_dim": hidden_dim_mixture, "output_dim": 3}
            it.tuning_wrapper(pool=pool, scenario=scenario, data_directory_name="mixture_data",
                              network_model_class=gt.FullyConnectedNetwork,
                              test_sample_prop=hp.test_sample_prop, epoch_vet=epoch_vet,
                              trial_index_vet=trial_index_vet, result_dict_name=result_dict_name,
                              network_model_class_kwargs=network_kwargs_dict,
                              sample_size_vet=hp.sample_size_vet, learning_rate=hp.learning_rate,
                              weights_or_radius_kwargs={"cut_off_radius": hp.alt_cut_off_radius})

            print(f"{scenario}, number_forward_layers: {number_forward_layers}, hidden_dim: {hidden_dim_mixture} "
                  f"finished.")


# Analysis results.
for mixture_result_dict_name in mixture_result_dict_name_vet:
    tuning_result_dict_name = mixture_result_dict_name + "_null"
    it.process_plot_epoch_kl_raw_dict(pool=pool, tuning_result_dict_name=tuning_result_dict_name,
                                      sample_size_vet=sample_size_vet, trial_index_vet=trial_index_vet)

for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    it.plot_loss_kl(scenario="alt", tuning_result_dict_name=mixture_result_dict_name_vet[4],
                    trial_index_vet=[0, 10, 49, 60],
                    sample_size=sample_size, end_epoch=epoch, start_epoch=0, plot_train_loss_boolean=True,
                    plot_kl_boolean=True, plot_test_loss_boolean=True)

# Full model, Alt, mixture_16_40_0.01_test_prop:0.1. Epoch[96, 30,  110,   74].
# Full model, Null, mixture_1_40_0.01_test_prop:0.1. Epoch[18,  33,  24, 36].

########
# Null #
########
# Full model on the null data
# 1 layer
number_forward_layers_vet = [1]
hidden_dim_mixture_vet = [40]
learning_rate_mixture = hp.learning_rate_mixture

epoch_vet = [400, 300, 150, 100]
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

mixture_result_dict_name_vet = []
for number_forward_layers in number_forward_layers_vet:
    for hidden_dim_mixture in hidden_dim_mixture_vet:
        result_dict_name = f"mixture_model_{number_forward_layers}_{hidden_dim_mixture}_{learning_rate_mixture}"
        mixture_result_dict_name_vet.append(result_dict_name)
        network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                               "hidden_dim": hidden_dim_mixture, "output_dim": 3}
        it.tuning_wrapper(pool=pool, scenario="null", data_directory_name="mixture_data",
                          network_model_class=gt.FullyConnectedNetwork,
                          number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=epoch_vet,
                          trial_index_vet=trial_index_vet, result_dict_name=result_dict_name,
                          network_model_class_kwargs=network_kwargs_dict,
                          sample_size_vet=hp.sample_size_vet, learning_rate=learning_rate_mixture,
                          weights_or_radius_kwargs={"cut_off_radius": hp.null_cut_off_radius})
        print(f"layer {number_forward_layers}, hidden_dim {hidden_dim_mixture} finished.")

for mixture_result_dict_name in mixture_result_dict_name_vet:
    tuning_result_dict_name = mixture_result_dict_name + "_null"
    it.process_plot_epoch_kl_raw_dict(pool=pool, tuning_result_dict_name=tuning_result_dict_name,
                                      sample_size_vet=sample_size_vet, trial_index_vet=trial_index_vet)

# Settings: mixture_reduced_model_12_160_0.01_null
# Mean kls are [0.34200882 0.33393823 0.15965524 0.13649723]
# Std of kls are [0.24092219 0.12274825 0.02776599 0.02406268]
# Median optimal epochs are [171.5  29.   45.   52.

for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    it.plot_loss_kl(scenario="alt", tuning_result_dict_name=mixture_result_dict_name_vet[1],
                    trial_index_vet=[0, 10, 49, 60],
                    sample_size=sample_size, end_epoch=epoch, start_epoch=0, plot_train_loss_boolean=True,
                    plot_kl_boolean=True, plot_test_loss_boolean=True)

##################################
# Fit null model on mixture data #
##################################
number_forward_layers_vet = [1, 4, 12, 16]
hidden_dim_mixture_vet = [40, 80, 160]
learning_rate_mixture = hp.learning_rate_mixture

epoch_vet = [500, 300, 150, 100]
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

mixture_result_dict_name_vet = []
for number_forward_layers in number_forward_layers_vet:
    for hidden_dim_mixture in hidden_dim_mixture_vet:
        result_dict_name = f"mixture_reduced_model_{number_forward_layers}_{hidden_dim_mixture}_{learning_rate_mixture}"
        mixture_result_dict_name_vet.append(result_dict_name)
        network_kwargs_dict = {"number_forward_layers": number_forward_layers, "input_dim": hp.dim_z,
                               "hidden_dim": hidden_dim_mixture, "output_dim": 2}
        it.tuning_wrapper(pool=pool, scenario="null", data_directory_name="mixture_data",
                          network_model_class=gt.FullyConnectedNetwork,
                          number_of_test_samples_vet=hp.number_of_test_samples_vet, epoch_vet=epoch_vet,
                          trial_index_vet=trial_index_vet, result_dict_name=result_dict_name,
                          network_model_class_kwargs=network_kwargs_dict,
                          sample_size_vet=hp.sample_size_vet, learning_rate=learning_rate_mixture,
                          weights_or_radius_kwargs={"cut_off_radius": hp.null_cut_off_radius})
        print(f"layer {number_forward_layers}, hidden_dim {hidden_dim_mixture} finished.")

# Analysis results.
for mixture_result_dict_name in mixture_result_dict_name_vet:
    tuning_result_dict_name = mixture_result_dict_name + "_null"
    it.process_plot_epoch_kl_raw_dict(pool=pool, tuning_result_dict_name=tuning_result_dict_name,
                                      sample_size_vet=sample_size_vet, trial_index_vet=trial_index_vet)

# Settings: mixture_reduced_model_12_160_0.01_null
# Mean kls are [0.15718039 0.19708511 0.1162377  0.09499046]
# Std of kls are [0.10966519 0.07473242 0.0221557  0.01745439]
# Median optimal epochs are [171.5  49.   68.5  66.5]

################################
# Tuning for true Ising model #
###############################
np.random.seed(hp.seed_index)
tf.random.set_seed(hp.seed_index)

epoch_vet = [50, 50, 20, 20]

true_result_dict_name = f"miss_2_50_{hp.learning_rate}"
true_one_layer_network_kwargs_dict = {"number_forward_layers": 2, "input_dim": hp.dim_z,
                                      "hidden_dim": 50, "output_dim": 3}
trial_index_vet = np.arange(120)
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
                    sample_size=sample_size, end_epoch=epoch, start_epoch=0, plot_train_loss_boolean=True,
                    plot_kl_boolean=True, plot_test_loss_boolean=True)

# The median optimal epochs are  1, 1, 8, 11.
# When sample size is 50 or 100, there seems to be overfitting problem.

pool.close()
pool.join()
