import numpy as np
import generate_train_fucntions as gt
import time
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import matplotlib.pyplot as plt

import pickle
import hyperparameters as hp


"""
import simulation_functions as sf
import time

# Ising simulation
start_time = time.time()

sf.oracle_ising_simulation_loop(scenario = "null", result_dict_name = "ising")
sf.oracle_ising_simulation_loop(scenario = "alt", result_dict_name = "ising")

print("Ising simulation takes %s seconds to finish." % (time.time() - start_time))

"""

"""
def ising_simulation_wrapper(simulation_index, scenario, sample_size, epoch):
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt", dtype = np.float32)
    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}_{simulation_index}.txt", dtype = np.float32)

    ising_training_pool_instance = gt.IsingTrainingPool(z_mat=z_mat, hidden_1_out_dim=hp.hidden_1_out_dim,
                                                        learning_rate=hp.learning_rate, buffer_size=hp.buffer_size,
                                                        batch_size=hp.batch_size, epoch=epoch)

    predicted_parameter_mat = ising_training_pool_instance.trainning(x_y_mat = x_y_mat)

    print(f"{scenario}: Sample size {sample_size} simulation {simulation_index} is done.")

    return (simulation_index, predicted_parameter_mat)


###########################
# Simulate under the null #
###########################
start_time_null = time.time()

ising_result_null_dict = dict()

for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet):
    pool = mp.Pool(processes=hp.process_number)
    simulation_indeepoch_vet = range(hp.simulation_times)
    pool_result_vet = pool.map(partial(ising_simulation_wrapper, sample_size=sample_size, scenario="null"),
                               simulation_indeepoch_vet)

    ising_result_null_dict[sample_size] = dict(pool_result_vet)

with open("./results/ising_result_null_dict.p", "wb") as fp:
    pickle.dump(ising_result_null_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

end_time_null = time.time() - start_time_null

##################################
# Simulate under the alternative #
##################################
start_time_alt = time.time()

ising_result_alt_dict = dict()

for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet):
    pool = mp.Pool(processes=hp.process_number)
    simulation_indeepoch_vet = range(hp.simulation_times)
    pool_result_vet = pool.map(partial(ising_simulation_wrapper,
                                       sample_size=sample_size, scenario="alt"),
                               simulation_indeepoch_vet)

    ising_result_alt_dict[sample_size] = dict(pool_result_vet)

with open("./results/ising_result_alt_dict.p", "wb") as fp:
    pickle.dump(ising_result_alt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

end_time_alt = time.time() - start_time_alt

print(f"Null take {end_time_null} to train.")
print(f"Alt takes {end_time_alt} to train.")
"""
class WrongIsingNetwork(tf.keras.Model):
    def __init__(self, input_dim, hidden_1_out_dim, hidden_2_out_dim,output_dim):
        super().__init__(input_dim, hidden_1_out_dim, output_dim)


        self.linear_1 = tf.keras.layers.Dense(
            units=hidden_1_out_dim,
            input_shape=(input_dim,)
        )

        self.linear_2 = tf.keras.layers.Dense(
            units=hidden_2_out_dim,
            input_shape=(hidden_1_out_dim,)
        )

        self.linear_3 = tf.keras.layers.Dense(
            units=output_dim,
            input_shape=(hidden_1_out_dim,)
        )

    def call(self, input):
        output = self.linear_1(input)
        output = tf.keras.activations.relu(output)
        output = self.linear_2(output)
        output = tf.keras.activations.elu(output)
        output = self.linear_3(output)
        output = tf.keras.activations.elu(output)
        return output

with open('./data/weights_dict.p', 'rb') as fp:
    weights_dict = pickle.load(fp)

scenario = "alt"
trails_vet = np.arange(4)
experiment_result_dict = dict()
for sample_size, epoch in zip(hp.sample_size_vet, hp.epoch_vet):
    sample_size_result_dict = dict()
    for trail in trails_vet:
        x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{trail}.txt", dtype=np.float32)
        z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}_{trail}.txt", dtype=np.float32)

        weights = weights_dict[sample_size][trail]
        true_network = gt.IsingNetwork(3, 3, 3)
        true_network.dummy_run()
        true_network.set_weights(weights)
        true_parameter_mat = true_network(z_mat)

        wrong_ising_network = WrongIsingNetwork(input_dim=3, hidden_1_out_dim=2, hidden_2_out_dim=2, output_dim=3)
        ising_tunning_instance = gt.IsingTunning(z_mat=z_mat, x_y_mat=x_y_mat, true_parameter_mat=true_parameter_mat,
                                                 ising_network=wrong_ising_network, batch_size = 100,
                                                 max_epoch = int(epoch * 1.2))

        sample_size_result_dict[trail] = ising_tunning_instance.trainning(True)

    experiment_result_dict[sample_size] = sample_size_result_dict

with open(f"./tunning/experiment_result_dict.p", "wb") as fp:
    pickle.dump(experiment_result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


###################
# Result analysis #
###################
def plot_loss_kl(experiment_result_dict, sample_size, start_epoch = 0,plot_loss = True, plot_kl = True,
                 plot_test_loss = True):
    """
    Assuming there are only four sample size we are simulating. We plot the RoC curve and save the plot under
    ./results/plots/{model_for_main_title}.png.

    :param experiment_result_dict: A dictionary which is the output of the fpr_tpr function.
    :param model_for_main_title: A string ('str' class) which we use to name the graph to save as
    {model_for_main_title}.png.

    :return:
    None
    """
    epoch_vet = np.arange(experiment_result_dict[sample_size][0]["loss_array"][0, :].shape[0])[start_epoch:]
    fig, ax = plt.subplots(4, 2)

    if not plot_kl and not plot_loss and not plot_test_loss:
        print("Nothing to print")
        return

    if plot_loss:
        ax[0, 0].plot(epoch_vet, experiment_result_dict[sample_size][0]["loss_array"][0, start_epoch:], label = "loss")
        ax[1, 0].plot(epoch_vet, experiment_result_dict[sample_size][1]["loss_array"][0, start_epoch:])
        ax[2, 0].plot(epoch_vet, experiment_result_dict[sample_size][2]["loss_array"][0, start_epoch:])
        ax[3, 0].plot(epoch_vet, experiment_result_dict[sample_size][3]["loss_array"][0, start_epoch:])

    if plot_kl:
        ax[0, 1].plot(epoch_vet, experiment_result_dict[sample_size][0]["loss_array"][1, start_epoch:])
        ax[1, 1].plot(epoch_vet, experiment_result_dict[sample_size][1]["loss_array"][1, start_epoch:])
        ax[2, 1].plot(epoch_vet, experiment_result_dict[sample_size][2]["loss_array"][1, start_epoch:])
        ax[3, 1].plot(epoch_vet, experiment_result_dict[sample_size][3]["loss_array"][1, start_epoch:])

    if plot_test_loss:
        ax[0, 0].plot(epoch_vet, experiment_result_dict[sample_size][0]["loss_array"][2, start_epoch:], label = "test")
        ax[1, 0].plot(epoch_vet, experiment_result_dict[sample_size][1]["loss_array"][2, start_epoch:])
        ax[2, 0].plot(epoch_vet, experiment_result_dict[sample_size][2]["loss_array"][2, start_epoch:])
        ax[3, 0].plot(epoch_vet, experiment_result_dict[sample_size][3]["loss_array"][2, start_epoch:])

    ax[0, 0].legend()

    fig.suptitle(f"Sample size {sample_size}")
    fig.show()
    fig.savefig(f"./tunning/plots/{sample_size}_loss_{plot_loss}_kl_{plot_kl}_test_{plot_test_loss}.png")


for sample_size in hp.sample_size_vet:
    plot_loss_kl(experiment_result_dict, sample_size, plot_loss=False)


