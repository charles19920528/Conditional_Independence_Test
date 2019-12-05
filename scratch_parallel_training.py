import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import pickle
import multiprocessing as mp
from generate_z import dim_z, sample_size_vet

####################
# Hyper parameters #
####################
hidden_1_out_dim = 3
# Number of times we run the simulation for each sample size
simulation_times = 1000
epoch_vet = [250, 250, 100, 90]

###########################
# Simulate under the null #
###########################
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

############################
# Parallel Computing setup #
############################
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
print("Num GPUs Available: ", len(tf.config.experimental.list_logical_devices('GPU')))
virtual_gpu_list = tf.config.experimental.list_logical_devices('GPU')

class parallelIsingSimulation():
    def __init__(self, ising_simulation_instance, simulation_times, virtual_gpu_list, gpu_index, process_indexd,
                 print_loss_boolean):
        self.ising_simulation_instance = ising_simulation_instance
        self.simulation_times = simulation_times
        self.virtual_gpu_list = virtual_gpu_list

    def sequential_simulation(self):
        sample_result_dict = dict()
        for j in np.arange(simulation_times):
            x_y_mat = self.ising_simulation_instance.generate_x_y_mat()
            sample_result_dict[j] = self.ising_simulation_instance.trainning(x_y_mat, print_loss_boolean=False)
            print("Training finished. Sample size : %d, simulation sample: %d" % (sample_size, j))

        return sample_result_dict

    def parallel_training(self):
        process_list = [mp.Process()]


output_queue = mp.Queue()
def addition(a, b, output_queue):
    print(f"computing {a} plus {b}")
    output_queue.put(a + b)

addition(1, 2, output_queue)

a_list = [1, 2, 3]
b_list = [4, 5, 6]
process_list = [mp.Process(target = addition, args = (a_list[i], b_list[i]) ) for i in range(3) ]

for process in process_list:
    process.start()

for process in process_list:
    process.join()

result_dict = dict()
for i, process in enumerate(process_list):
    result_dict[i] = process



def parallel_ising_simulation(ising_simulation_instance, simulation_times, virtual_gpus_list, gpu_index,
                              process_indexd, print_loss_boolean):

    tf.config.experimental.set_visible_devices(devices = virtual_gpus_list)

    sample_result_dict = dict()
    for j in np.arange(simulation_times):
        x_y_mat = ising_simulation_instance.generate_x_y_mat()
        sample_result_dict[j] = ising_simulation_instance.trainning(x_y_mat, print_loss_boolean=False)
        print("Process %d: Training finished. Sample size : %d, simulation sample: %d" %
              (sample_size, j, process_indexd))

    return sample_result_dict









scratch_null_result_dict = dict()
for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    z_mat = np.loadtxt("./data/z_%d.txt" % sample_size, dtype="float32")

    ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=null_network_generate, null_boolean=True,
                                          hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                          batch_size=50, epoch=epoch)
    sample_result_dict = dict()
    for j in np.arange(simulation_times):
        x_y_mat = ising_simulation.generate_x_y_mat()
        sample_result_dict[j] = ising_simulation.trainning(x_y_mat, print_loss_boolean=False)
        print("Training finished. Sample size : %d, simulation sample: %d" % (sample_size, j))
    scratch_null_result_dict[sample_size] = sample_result_dict


with open("./results/scratch_null_result_dict.p", "wb") as fp:
    pickle.dump(scratch_null_result_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)