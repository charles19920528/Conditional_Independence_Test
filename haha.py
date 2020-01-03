import tensorflow as tf
import numpy as np
import time
import itertools
import generate_train_fucntions as gt
import hyperparameters as hp



z_mat = tf.data.TextLineDataset("./data/z_mat/z_mat_30.txt")
x_y_mat = tf.data.TextLineDataset("./data/null/x_y_mat_30_0.txt")

# Helper functions to read data using tensorflow dataset module.
def parse_fnc(line):
    string_vals = tf.strings.split([line]).values
    return tf.strings.to_number(string_vals, tf.float32)
z_mat = z_mat.map(parse_fnc)
x_y_mat = x_y_mat.map(parse_fnc)

def tf_load_z_mat(sample_size_tensor):
    z_file_format = tf.strings.format("./data/z_mat/z_mat_{}.txt", inputs=sample_size_tensor)
    z_mat = tf.data.TextLineDataset(z_file_format)
    z_mat = z_mat.map(parse_fnc)

    return z_mat

z_mat = tf_load_z_mat(tf.constant(30))
# print(list(z_mat.take(1)))

def tf_load_x_y_mat_null(sample_size_tensor, simulation_times_tensor):
    x_y_file_format = tf.strings.format("./data/null/x_y_mat_{}_{}.txt", inputs=(sample_size_tensor,
                                                                               simulation_times_tensor))
    x_y_mat = tf.data.TextLineDataset(x_y_file_format)
    x_y_mat = x_y_mat.map(parse_fnc)

    return x_y_mat

# x_y_mat = tf_load_x_y_mat_null(tf.constant(30), tf.constant(1))
# print(list(x_y_mat.take(1)))

def tf_load_x_y_mat_alt(sample_size_tensor, simulation_times_tensor):
    x_y_file_format = tf.strings.format("./data/alt/x_y_mat_{}_{}.txt", inputs=(sample_size_tensor,
                                                                               simulation_times_tensor))
    x_y_mat = tf.data.TextLineDataset(x_y_file_format)
    x_y_mat = x_y_mat.map(parse_fnc)

    return x_y_mat

# x_y_mat = tf_load_x_y_mat_alt(tf.constant(30), tf.constant(1))
# print(list(x_y_mat.take(1)))

def dt_iterator():
    for z, x_y in zip(z_mat, x_y_mat):
        yield (z.numpy(), x_y.numpy())





sample_size_vet = [30]
epoch_vet = [1]

# Simulation under the null.
null_result_dict = dict()
for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    sample_size_tensor = tf.constant(sample_size)
    epoch_tensor = tf.constant(epoch)

    z_mat = tf_load_z_mat(sample_size_tensor = sample_size_tensor)

    ising_training_instance = gt.IsingTraining(z_mat = z_mat, hidden_1_out_dim = hp.hidden_1_out_dim,
                                              learning_rate = 0.005, buffer_size = 1024, batch_size = 30,
                                              epoch = epoch)

    sample_size_result_variable = tf.Variable( tf.zeros((sample_size, 3 * hp.simulation_times)) )
    @tf.function
    def parallel_training(IsingTraining_instance, simulation_times):
        i = tf.constant(0, dtype=tf.int32)
        while tf.less(i, simulation_times):
            x_y_mat = tf_load_x_y_mat_null(sample_size_tensor = sample_size_tensor, simulation_times_tensor = i)

            predicted_parameter_mat = IsingTraining_instance.trainning(x_y_mat)
            sample_size_result_variable[:, (i * 3) : (i * 3 + 3)].assign(predicted_parameter_mat)

            train_message_format = tf.strings.format("Simulation {} is finished", input = i)
            tf.print(train_message_format)

            i += 1

    parallel_training(IsingTraining_instance = ising_training_instance, simulation_times = 3)

    null_result_dict[sample_size] = sample_size_result_variable.value()


x = tf.Variable(tf.eye(4))
type(x.value().numpy())


































#######################################################################################################################

z_mat = tf.data.TextLineDataset("./data/z_mat/z_mat_30.txt")
x_y_mat = tf.data.TextLineDataset("./data/null/x_y_mat_30_0.txt")

def parse_fnc(line):
    string_vals = tf.strings.split([line]).values
    return tf.strings.to_number(string_vals, tf.float32)
z_mat = z_mat.map(parse_fnc)
x_y_mat = x_y_mat.map(parse_fnc)



def dt_iterator():
    for z, x_y in zip(z_mat, x_y_mat):
        yield (z.numpy(), x_y.numpy())

test = dt_iterator()
t = next(test)

concaternated_dataset = tf.data.Dataset.from_generator(dt_iterator,
                                                       output_types = (tf.float32, tf.float32))





z_mat = np.loadtxt("./data/z_mat/z_mat_%d.txt" % 30, dtype="float32")
ising_training_istance = gt.IsingTraining(z_mat = z_mat, hidden_1_out_dim = hp.hidden_1_out_dim, learning_rate = 0.005,
                                          buffer_size = 1000, batch_size = 30, epoch = 1)
sample_size_result_variable = tf.Variable(tf.zeros( (30, 3 * hp.simulation_times) ) )
x_y_mat = np.loadtxt(f"./data/null/x_y_mat_30_0.txt")
result = ising_training_istance.trainning(x_y_mat)
sample_size_result_variable[:, (0 * 3) : (0 * 3 + 3)].assign(result)


sample_size_vet = [30]
epoch_vet = [1]

null_result_dict = dict()
for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    z_mat = np.loadtxt("./data/z_mat/z_mat_%d.txt" % sample_size, dtype="float32")
    ising_training_instance = gt.IsingTraining(z_mat = z_mat, hidden_1_out_dim = hp.hidden_1_out_dim,
                                              learning_rate = 0.005, buffer_size = 1000, batch_size = 30,
                                              epoch = epoch)

    sample_size_result_variable = tf.Variable( tf.zeros((sample_size, 3 * hp.simulation_times)) )
    @tf.function
    def parallel_training(IsingTraining_instance, scenario, sample_size, simulation_times):
        i = tf.constant(0, dtype=tf.int32)
        while tf.less(i, simulation_times):
            print(i)
            x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{i}.txt")
            predicted_parameter_mat = IsingTraining_instance.trainning(x_y_mat, False)
            sample_size_result_variable[:, (i * 3) : (i * 3 + 3)].assign(predicted_parameter_mat)

            print(f"Simulation {i} is finished")
            i += 1

    parallel_training(IsingTraining_instance = ising_training_istance, scenario = "null", sample_size = sample_size,
                      simulation_times = 3)

print(sample_size_result_variable[:, : 9])



"""
result = tf.Variable(np.zeros([10], dtype=np.int32))

@tf.function
def run_graph():
  i = tf.constant(0, dtype=tf.int32)
  while tf.less(i, 10):
      result[i].assign(i)  # Performance may require tuning here.
      formatted = tf.strings.format("./data/null/x_y_mat_30_{}.txt", i)
      tf.print(formatted)
      test = tf.data.TextLineDataset(formatted)

      time.sleep(1)
      i += 1



run_graph()
print(result.read_value())
"""


