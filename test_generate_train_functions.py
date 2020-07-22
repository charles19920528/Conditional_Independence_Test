import generate_train_functions as gt
import tensorflow as tf
import numpy as np
from multiprocessing import Pool
import hyperparameters as hp

tf.random.set_seed(0)
np.random.seed(0)


# Test the IsingNetwork class.
null_ising_network = gt.IsingNetwork(input_dim=3, hidden_1_out_dim=3, output_dim=2)
null_ising_network.dummy_run()
alt_ising_network = gt.IsingNetwork(input_dim=3, hidden_1_out_dim=3, output_dim=3)
alt_ising_network.dummy_run()


# Test the pmf_collection function.
null_parameter_mat = tf.constant([[-1, 0], [1, 2], [-3, -2]])
null_pmf_collection_mat = gt.pmf_collection(parameter_mat=null_parameter_mat)

alt_parameter_mat = tf.constant([[-1, 0, 1], [1, 2, 3], [-3, 1, -2]])
alt_pmf_collection_mat = gt.pmf_collection(parameter_mat=alt_parameter_mat)


# Test log_ising_likelihood function.
x_y_mat = tf.constant([[1, 1], [1, -1], [-1, -1]])
null_log_likelihood = gt.log_ising_likelihood(x_y_mat = x_y_mat, parameter_mat = null_parameter_mat)
alt_log_likelihood = gt.log_ising_likelihood(x_y_mat = x_y_mat, parameter_mat = alt_parameter_mat)


# Test the generate_x_y_mat function.
p_mat = tf.constant([[0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0], [0, 1, 0, 0]])
x_y_mat = gt.generate_x_y_mat(p_mat)


# Test the generate_x_y_mat_ising function.
z_mat = tf.random.normal(shape=(4, 3))
null_x_y_mat = gt.generate_x_y_mat_ising(ising_network=null_ising_network, z_mat=z_mat)
alt_x_y_mat = gt.generate_x_y_mat_ising(ising_network=alt_ising_network, z_mat=z_mat)


# Test the data_generate_network function.
null_network_generate, alt_network_generate, weights_list = \
    gt.data_generate_network(weights_distribution_string="glorot_normal")
null_network_generate(z_mat)
null_network_generate.get_weights()


# Test the conditional_pmf_collection_mixture function.
null_conditional_pmf_mat = gt.conditional_pmf_collection_mixture(z_mat=z_mat, is_null_boolean=True,
                                                                 cut_off_radius=0.5)
null_conditional_pmf_mat = gt.conditional_pmf_collection_mixture(z_mat=z_mat, is_null_boolean=True,
                                                                 cut_off_radius=10)
alt_conditional_pmf_mat = gt.conditional_pmf_collection_mixture(z_mat=z_mat, is_null_boolean=False,
                                                                 cut_off_radius=0.5)
alt_conditional_pmf_mat = gt.conditional_pmf_collection_mixture(z_mat=z_mat, is_null_boolean=False,
                                                                 cut_off_radius=10)


# Test kl_divergence function.
kl_divergence = gt.kl_divergence_ising(true_parameter_mat=null_parameter_mat, predicted_parameter_mat=alt_parameter_mat,
                                       isAverage=True)
kl_divergence = gt.kl_divergence_ising(true_parameter_mat=null_parameter_mat, predicted_parameter_mat=alt_parameter_mat,
                                       isAverage=False)


# Test the IsingTrainingTuning class.
data_directory_name = "mixture_data"
scenario = "null"
sample_size = 50
trial_index = 10

x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt",
                     dtype=np.float32)
z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)
network_model_class_kwargs = {"number_forward_layers": 1, "input_dim": 3,
                              "hidden_dim": 3, "output_dim": 3}

network_tt_instance = gt.NetworkTrainingTuning(z_mat=z_mat, x_y_mat=x_y_mat,
                                               network_model_class=gt.FullyConnectedNetwork,
                                               network_model_class_kwargs=network_model_class_kwargs, epoch=21)
loss_kl_array = network_tt_instance.tuning(print_loss_boolean=True, is_null_boolean=True, number_of_test_samples=5,
                                           cut_off_radius=hp.null_cut_off_radius)

network_tt_instance.train_compute_test_statistic(print_loss_boolean=True, number_of_test_samples=5)

pool = Pool(processes=4)
network_tt_instance.bootstrap(pool=pool, number_of_bootstrap_samples=5)

pool.close()
pool.join()



#
# np.random.seed(0)
# pool = Pool(processes=hp.process_number)
#
# z_mat = np.random.normal(size=(4, 3))
# network_model_class_kwargs = {"number_forward_layers": 1, "input_dim": 3,
#                               "hidden_dim": 3, "output_dim": 3}
# network_net_size=5
# number_of_nets=10
# gt.argmax_gaussian_process_one_trial_one_net(_=1, z_mat=z_mat, network_model_class=gt.FullyConnectedNetwork,
#                                              network_model_class_kwargs=network_model_class_kwargs,
#                                              network_net_size=network_net_size)
#
#
# # Test ising_test_statistic_distribution_one_trial function.
# gt.argmax_gaussian_process_one_trial(pool=pool, z_mat=z_mat, network_model_class=gt.FullyConnectedNetwork,
#                                      network_model_class_kwargs=network_model_class_kwargs,
#                                      network_net_size=hp.network_net_size, number_of_nets=number_of_nets)
#
# pool.close()
# pool.join()








# Test ising_test_statistic_distribution_one_trial_one_net function.
# np.random.seed(0)
# pool = Pool(processes=hp.process_number)
#
# z_mat = np.random.normal(size=(4, 3))
# network_model_class_kwargs = {"number_forward_layers": 1, "input_dim": 3,
#                               "hidden_dim": 3, "output_dim": 3}
# network_net_size=5
# number_of_nets=10
# gt.argmax_gaussian_process_one_trial_one_net(_=1, z_mat=z_mat, network_model_class=gt.FullyConnectedNetwork,
#                                              network_model_class_kwargs=network_model_class_kwargs,
#                                              network_net_size=network_net_size)
#
#
# # Test ising_test_statistic_distribution_one_trial function.
# gt.argmax_gaussian_process_one_trial(pool=pool, z_mat=z_mat, network_model_class=gt.FullyConnectedNetwork,
#                                      network_model_class_kwargs=network_model_class_kwargs,
#                                      network_net_size=hp.network_net_size, number_of_nets=number_of_nets)
#
# pool.close()
# pool.join()






