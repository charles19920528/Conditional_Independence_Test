import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import pickle
from hyperparameters import hidden_1_out_dim, simulation_times, dim_z, sample_size_vet, epoch_vet

# Only run on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##################################
# Simulate under the alternative #
##################################
# Create the alternative network
tf.random.set_seed(1)
alt_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 3)
alt_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape=(dim_z, hidden_1_out_dim), mean=1, stddev=1)
linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = tf.random.normal(shape=(hidden_1_out_dim, 3), mean=1, stddev=1)
linear_2_bias_array = tf.zeros(shape=(3,))

alt_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])

alternative_result_dict = dict()
for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    z_mat = np.loadtxt("./data/z_mat_%d.txt" % sample_size, dtype="float32")

    ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=alt_network_generate, null_boolean=False,
                                          hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                          batch_size=50, epoch=epoch)
    sample_result_dict = dict()
    for j in np.arange(simulation_times):
        x_y_mat = ising_simulation.generate_x_y_mat()
        sample_result_dict[j] = ising_simulation.trainning(x_y_mat, print_loss_boolean=False)
        print("Training finished. Sample size : %d, simulation sample: %d" % (sample_size, j))
    alternative_result_dict[sample_size] = sample_result_dict

with open("./results/alternative_result_dict.p", "wb") as fp:
    pickle.dump(alternative_result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

