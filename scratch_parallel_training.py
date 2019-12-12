import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
from hyperparameters import hidden_1_out_dim, simulation_times, sample_size_vet, epoch_vet
import pickle
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dim_z = 3
simulation_times = 20

sample_size_vet = [30]
# Training epochs for samples sizes in the sample_size_vet
epoch_vet = [10]
#########################
# Create 8 virtual GPUs #
#########################
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
print("Num virtual GPUs Available: ", len(tf.config.experimental.list_logical_devices('GPU')))
virtual_gpu_list = tf.config.experimental.list_logical_devices('GPU')
"""


# Create null network
tf.random.set_seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape=(dim_z, hidden_1_out_dim), mean=1, stddev=1)
linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = tf.random.normal(shape=(hidden_1_out_dim, 2), mean=1, stddev=1)
linear_2_bias_array = tf.zeros(shape=(2,))

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])

########## My implementation
condition_function = lambda i: i < simulation_times
def loop_function(iteration):
    x_y_mat = np.loadtxt(f"./data/x_y_mat_{sample_size}_{iteration}.txt", dtype="float32")
    sample_result_dict[iteration.numpy()] = ising_simulation.trainning(x_y_mat, print_loss_boolean=False)
    print("Training finished. Sample size : %d, simulation sample: %d" % (sample_size, iteration))
    iteration += 1
    return (iteration, )

scratch_null_result_dict = dict()
for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    z_mat = np.loadtxt("./data/z_%d.txt" % sample_size, dtype="float32")

    ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=null_network_generate , null_boolean=True,
                                          hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                          batch_size=50, epoch=epoch)
    sample_result_dict = dict()
    loop_var = tf.constant(0)
    start_time = time.time()
    tf.while_loop(cond = condition_function, body = loop_function, loop_vars = [loop_var], parallel_iterations=10)
    scratch_null_result_dict[sample_size] = sample_result_dict

with open("./results/null_result_dict.p", "wb") as fp:
    pickle.dump(scratch_null_result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


iteration = tf.constant(0)
c = lambda i: tf.less(i, 10)
def print_fun(iteration):
    print(iteration)
    iteration+=1
    return (iteration,)
r = tf.while_loop(c, print_fun, [iteration])





start_time = time.time()

iteration = tf.constant(0)
c = lambda i: tf.less(i, 1010)
b = lambda i: (tf.add(i, 1),)
def print_fun(iteration):
    print(f"This is iteration {iteration}")
    iteration+=1
    return (iteration,)
r = tf.while_loop(c, b, [iteration], parallel_iterations=1)

print("--- %s seconds ---" % (time.time() - start_time))




