import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import pickle
import time
from hyperparameters import hidden_1_out_dim, dim_z

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
'''
scratch_null_result_dict = dict()
for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    z_mat = np.loadtxt("./data/null/z_%d.txt" % sample_size, dtype="float32")

    ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=null_network_generate , null_boolean=True,
                                          hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                          batch_size=50, epoch=epoch)
    sample_result_dict = dict()

    condition_function = lambda i: i < simulation_times


    start_time = time.time()
    run_graph()
    print("--- %s seconds ---" % (time.time() - start_time))
    scratch_null_result_dict[sample_size] = sample_result_dict


with open("./results/null_result_dict.p", "wb") as fp:
    pickle.dump(scratch_null_result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
'''

result = np.zeros(10)
iteration = tf.constant(0)
c = lambda i: tf.less(i, 10)
def print_fun(iteration):
    result[iteration] = iteration
    iteration += 1
    return (iteration,)


start_time = time.time()
r = tf.while_loop(c, print_fun, [iteration])
print("--- %s seconds ---" % (time.time() - start_time))


@tf.function
def run_graph():
    iteration = tf.constant(0)
    tf.while_loop(c, print_fun, [iteration])

run_graph()



"""

@tf.function
def run_graph():
    iteration = tf.constant(0)
    tf.while_loop(c, print_fun, [iteration], parallel_iterations=10)

start_time = time.time()
run_graph()
print("--- %s seconds ---" % (time.time() - start_time))
"""
