import tensorflow as tf
import generate_train_fucntions as gt
import hyperparameters as hp

#####################################################################################################
# Still work on it. Haven't been able to make it work.



##################################################################################################33
sample_size_vet = [30]
epoch_vet = [1]
simulation_times = 3
# Simulation under the null.
null_result_dict = dict()
for sample_size, epoch in zip(sample_size_vet, epoch_vet):
    sample_size_tensor = tf.constant(sample_size)
    epoch_tensor = tf.constant(epoch)

    z_dataset = gt.tf_load_z_dataset(sample_size_tensor = sample_size_tensor)


    sample_size_result_variable = tf.Variable( tf.zeros((sample_size, 3 * simulation_times)) )
#    @tf.function
    def parallel_training(simulation_times):
        i = tf.constant(0, dtype=tf.int32)
        f = tf.strings.format("Simulation {} started", inputs = i)
        tf.print(f)
        while tf.less(i, simulation_times):
            x_y_dataset = gt.tf_load_x_y_dataset_null(sample_size_tensor = sample_size_tensor, simulation_times_tensor = i)


            ising_training_instance = gt.IsingTraining_tf_function(z_dataset = z_dataset, x_y_dataset=x_y_dataset,
                                                                   z_dim = hp.dim_z,
                                                       hidden_1_out_dim = hp.hidden_1_out_dim, sample_size=sample_size,
                                                       learning_rate = hp.learning_rate, buffer_size=hp.buffer_size,
                                                       batch_size = 5, epoch=epoch)


            predicted_parameter_mat = ising_training_instance.trainning()
            sample_size_result_variable[:, (i * 3) : (i * 3 + 3)].assign(predicted_parameter_mat)

            train_message_format = tf.strings.format("Simulation {} is finished", inputs = i)
            tf.print(train_message_format)

            i += 1

    parallel_training(simulation_times = simulation_times)
    null_result_dict[sample_size] = sample_size_result_variable.value()

    print(f"Simulation on sample size {sample_size} is finished")





























