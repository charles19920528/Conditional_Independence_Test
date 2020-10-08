from functools import partial
import pickle
import os
import numpy as np
import tensorflow as tf
import hyperparameters as hp
import generate_train_functions as gt
from multiprocessing import Pool
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def ising_bootstrap_one_trial(_, z_mat, x_y_mat, train_indices_vet, test_indices_vet,
                              network_model_class, network_model_class_kwargs, buffer_size, batch_size, learning_rate,
                              full_model_max_epoch, reduced_model_max_epoch):
    # Train null network.
    reduced_model_class_kwargs = network_model_class_kwargs.copy()
    reduced_model_class_kwargs["output_dim"] = 2
    reduced_train_ds = tf.data.Dataset.from_tensor_slices((z_mat[train_indices_vet, :], x_y_mat[train_indices_vet, :]))
    reduced_train_ds = reduced_train_ds.shuffle(buffer_size).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    reduced_model = network_model_class(**reduced_model_class_kwargs)
    epoch = 0
    while epoch < reduced_model_max_epoch:
        _ = gt.train_network(train_ds=reduced_train_ds, optimizer=optimizer, network_model=reduced_model)
        epoch += 1

    # Resample
#    fitted_par_mat = reduced_model(z_mat[train_indices_vet, :])
#    fitted_train_p_mat = gt.pmf_collection(fitted_par_mat)
#    new_train_x_y_mat = gt.generate_x_y_mat(fitted_train_p_mat)
    new_train_x_y_mat = gt.generate_x_y_mat_ising(ising_network=reduced_model, z_mat=z_mat[train_indices_vet, :])
    train_ds = tf.data.Dataset.from_tensor_slices((z_mat[train_indices_vet, :], new_train_x_y_mat))
    train_ds = train_ds.shuffle(buffer_size).batch(batch_size)

    # Train the network
    network_model = network_model_class(**network_model_class_kwargs)
    epoch = 0
    while epoch < full_model_max_epoch:
        _ = gt.train_network(train_ds=train_ds, optimizer=optimizer, network_model=network_model)
        epoch += 1

    predicted_parameter_mat = network_model(z_mat)
    jxy_squared_vet = np.square(predicted_parameter_mat[:, 2])
    result_dict = {"train_test_statistic": np.mean(jxy_squared_vet[train_indices_vet]),
                   "test_test_statistic": np.mean(jxy_squared_vet[test_indices_vet])}

    return result_dict


def ising_bootstrap_method(pool, trial_index, sample_size, scenario, data_directory_name,
                           ising_simulation_result_dict_name, network_model_class, network_model_class_kwargs,
                           number_of_bootstrap_samples, full_model_max_epoch, reduced_model_max_epoch,
                           batch_size=hp.batch_size, buffer_size=hp.buffer_size,
                           learning_rate=hp.learning_rate):
    with open(f'results/result_dict/{data_directory_name}/{ising_simulation_result_dict_name}_{scenario}_result_'
              f'dict.p', 'rb') as fp:
        ising_simulation_loop_result_dict = pickle.load(fp)

    z_mat = np.loadtxt(f"./data/{data_directory_name}/z_mat/z_mat_{sample_size}_{trial_index}.txt", dtype=np.float32)
    x_y_mat = np.loadtxt(f"./data/{data_directory_name}/{scenario}/x_y_mat_{sample_size}_{trial_index}.txt",
                         dtype=np.float32)
    train_indices_vet = ising_simulation_loop_result_dict[sample_size][trial_index]["train_indices_vet"]
    test_indices_vet = ising_simulation_loop_result_dict[sample_size][trial_index]["test_indices_vet"]
    train_test_statistic = ising_simulation_loop_result_dict[sample_size][trial_index]["train_test_statistic"]
    test_test_statistic = ising_simulation_loop_result_dict[sample_size][trial_index]["test_test_statistic"]

    bootstrap_test_statistic_dict_vet = pool.map(partial(ising_bootstrap_one_trial,
                                                         z_mat=z_mat, train_indices_vet=train_indices_vet,
                                                         test_indices_vet=test_indices_vet,
                                                         network_model_class=network_model_class,
                                                         network_model_class_kwargs=network_model_class_kwargs,
                                                         buffer_size=buffer_size, batch_size=batch_size,
                                                         learning_rate=learning_rate, x_y_mat=x_y_mat,
                                                         full_model_max_epoch=full_model_max_epoch,
                                                         reduced_model_max_epoch=reduced_model_max_epoch),
                                                 np.arange(number_of_bootstrap_samples))
    train_test_statistic_vet = np.zeros(number_of_bootstrap_samples)
    test_test_statistic_vet = np.zeros(number_of_bootstrap_samples)
    for i, trial_dict in enumerate(bootstrap_test_statistic_dict_vet):
        train_test_statistic_vet[i] = trial_dict["train_test_statistic"]
        test_test_statistic_vet[i] = trial_dict["test_test_statistic"]

    train_p_value = sum(train_test_statistic_vet > train_test_statistic) / number_of_bootstrap_samples
    test_p_value = sum(test_test_statistic_vet > test_test_statistic) / number_of_bootstrap_samples
    result_dict = {"train_p_value": train_p_value, "test_p_value": test_p_value,
                   "train_test_statistic_vet": train_test_statistic_vet,
                   "test_test_statistic_vet":  test_test_statistic_vet}

    return result_dict


def ising_bootstrap_loop(pool, scenario, data_directory_name, ising_simulation_result_dict_name, result_dict_name,
                         trial_index_vet, network_model_class,
                         network_model_class_kwargs_vet, number_of_bootstrap_samples, full_model_max_epoch_vet,
                         reduced_model_max_epoch_vet,
                         learning_rate=hp.learning_rate, buffer_size=hp.buffer_size, batch_size=hp.batch_size,
                         sample_size_vet=hp.sample_size_vet):
    result_dict = {}
    for sample_size, network_model_class_kwargs, \
        full_model_max_epoch, reduced_model_max_epoch in zip(sample_size_vet, network_model_class_kwargs_vet,
                                                             full_model_max_epoch_vet, reduced_model_max_epoch_vet):
        sample_size_result_dict = {}
        for trial_index in trial_index_vet:
            trial_result_dict = \
                ising_bootstrap_method(pool=pool, trial_index=trial_index, sample_size=sample_size, scenario=scenario,
                                       data_directory_name=data_directory_name,
                                       ising_simulation_result_dict_name=ising_simulation_result_dict_name,
                                       network_model_class=network_model_class,
                                       network_model_class_kwargs=network_model_class_kwargs,
                                       number_of_bootstrap_samples=number_of_bootstrap_samples,
                                       full_model_max_epoch=full_model_max_epoch,
                                       reduced_model_max_epoch=reduced_model_max_epoch,
                                       batch_size=batch_size, buffer_size=buffer_size, learning_rate=learning_rate)
            sample_size_result_dict[trial_index] = trial_result_dict
            print(f"Bootstrap, data: {data_directory_name}, scenario: {scenario}, sample_size: {sample_size}, "
                  f"trial_index: {trial_index} finished.")
            print(f"Train P-value is {trial_result_dict['train_p_value']}")
            print(f"Test P-value is {trial_result_dict['test_p_value']}")

        result_dict[sample_size] = sample_size_result_dict

    with open(f"./results/result_dict/{data_directory_name}/{result_dict_name}_{scenario}_result_dict.p", "wb") as fp:
        pickle.dump(result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


pool = Pool(processes=hp.process_number)

# Misspecified architecture on Ising data
for scenario in ["null", "alt"]:
    nfl_hd_vet = [(10, 100)]
    for nfl_hd in nfl_hd_vet:
        number_forward_layers, hidden_dim = nfl_hd

        ising_network_model_class_kwargs = {"number_forward_layers": number_forward_layers,
                                            "input_dim": hp.dim_z, "hidden_dim": hidden_dim, "output_dim": 3}
        ising_network_model_class_kwargs_vet = [ising_network_model_class_kwargs for _ in range(len(hp.sample_size_vet))][
                                               0:2]

        ising_bootstrap_loop(pool=pool, scenario=scenario, data_directory_name="ising_data",
                             ising_simulation_result_dict_name="ising_data_true_architecture",
                             result_dict_name=f"bootstrap_refit_reduced_nfl:{number_forward_layers}_hd:{hidden_dim}_500",
                             trial_index_vet=np.arange(200), network_model_class=gt.FullyConnectedNetwork,
                             network_model_class_kwargs_vet=ising_network_model_class_kwargs_vet,
                             number_of_bootstrap_samples=hp.number_of_boostrap_samples,
                             full_model_max_epoch_vet=[hp.ising_epoch_vet[2]],
                             reduced_model_max_epoch_vet=[hp.ising_epoch_vet[2]],
                             sample_size_vet=[hp.sample_size_vet[2]])

        print(f"{nfl_hd} finished.")


#######################################################
mixture_result_dict_name = f"mixture_data_{hp.mixture_number_forward_layer}_{hp.mixture_hidden_dim}"
mixture_network_model_class_kwargs = {"number_forward_layers": hp.mixture_number_forward_layer,
                                      "input_dim": hp.dim_z, "hidden_dim": hp.mixture_hidden_dim, "output_dim": 3}
mixture_network_model_class_kwargs_vet = [mixture_network_model_class_kwargs for _ in range(len(hp.sample_size_vet))]


ising_bootstrap_loop(pool=pool, scenario="null", data_directory_name="mixture_data",
                     ising_simulation_result_dict_name=mixture_result_dict_name,
                     result_dict_name="bootstrap_refit_reduced_mixture_50_100",
                     trial_index_vet=np.arange(100), network_model_class=gt.FullyConnectedNetwork,
                     network_model_class_kwargs_vet=mixture_network_model_class_kwargs_vet,
                     number_of_bootstrap_samples=hp.number_of_boostrap_samples,
                     full_model_max_epoch_vet=hp.mixture_epoch_vet[0:2],
                     reduced_model_max_epoch_vet=hp.reduced_model_epoch_vet[0:2],
                     sample_size_vet=hp.sample_size_vet[0:2])

pool.close()
pool.join()