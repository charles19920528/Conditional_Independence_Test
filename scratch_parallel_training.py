import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import pickle
import multiprocessing as mp
from generate_z import dim_z, sample_size_vet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
####################
# Hyper parameters #
####################
hidden_1_out_dim = 3
# Number of times we run the simulation for each sample size
simulation_times = 1000
epoch_vet = [250, 250, 100, 90]

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

#virtual_gpu_name_list = [ f"/gpu:{x}" for x in range(len(virtual_gpu_list))]


class parallelIsingSimulation():
    def __init__(self, ising_simulation_instance, simulation_times, virtual_gpu_name_list, number_of_process):
        self.ising_simulation_instance = ising_simulation_instance
        self.simulation_times = simulation_times
        self.virtual_gpu_name_list = virtual_gpu_name_list
        self.number_of_process = number_of_process
        self.output_queue = mp.Queue()

    def sequential_simulation(self, print_loss_boolean):
        """
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

        strategy = tf.distribute.MirroredStrategy(devices = [virtual_gpu_name])

        with strategy.scope():
            sample_result_dict = dict()
            for j in np.arange(self.simulation_times):
                x_y_mat = self.ising_simulation_instance.generate_x_y_mat()
                sample_result_dict[j] = self.ising_simulation_instance.trainning(x_y_mat, print_loss_boolean)
                print("Training finished. Sample size : %d, simulation sample: %d" % (x_y_mat.shape[0], j))

            self.output_queue.put(sample_result_dict)
        """
        sample_result_dict = dict()
        for j in np.arange(self.simulation_times):
            x_y_mat = self.ising_simulation_instance.generate_x_y_mat()
            sample_result_dict[j] = self.ising_simulation_instance.trainning(x_y_mat, print_loss_boolean)
            print("Training finished. Sample size : %d, simulation sample: %d" % (x_y_mat.shape[0], j))

        self.output_queue.put(sample_result_dict)


    def parallel_training(self, print_loss_boolean):
        """
        process_list = [mp.Process(target = self.sequential_simulation,
                                   args = (print_loss_boolean, self.virtual_gpu_name_list[i]))
                        for i in range(self.number_of_process)]
        """
        process_list = [mp.Process(target=self.sequential_simulation, args=[print_loss_boolean])
                        for i in range(self.number_of_process)]

        for process in process_list:
            process.start()

        for process in process_list:
            process.join()

z_mat = np.loadtxt("./data/z_%d.txt" % 30, dtype="float32")

ising_simulation = gt.IsingSimulation(z_mat=z_mat, true_network=null_network_generate, null_boolean=True,
                                          hidden_1_out_dim=hidden_1_out_dim, learning_rate=0.005, buffer_size=1000,
                                          batch_size=50, epoch=20)
parallel_ising_simulation = parallelIsingSimulation(ising_simulation, 1, 1024, 3)
parallel_ising_simulation.sequential_simulation(False)
parallel_ising_simulation.parallel_training(False)
parallel_ising_simulation.output_queue.qsize()

"""
output_queue = mp.Queue()
def addition(a, b, output_queue):
    print(f"computing {a} plus {b}")
    output_queue.put(a + b)


a_list = [1, 2, 3]
b_list = [4, 5, 6]
process_list = [mp.Process(target = addition, args = (a_list[i], b_list[i], output_queue) ) for i in range(3) ]

for process in process_list:
    process.start()

for process in process_list:
    process.join()

result_dict = dict()
[output_queue.get() for _ in process_list]
for i, process in enumerate(process_list):
    result_dict[i] = process
"""







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