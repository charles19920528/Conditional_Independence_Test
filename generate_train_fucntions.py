import tensorflow as tf
# Numpy is for the generate_x_y_mat_ising method and trainning method in the IsingSimulation class.
# Numpy is to solve process hung issue.
import numpy as np
import hyperparameters as hp

##################################
# Functions for data generation. #
##################################
class IsingNetwork(tf.keras.Model):
    # We consider this network as the true network.
    def __init__(self, input_dim, hidden_1_out_dim, output_dim):
        super().__init__(input_dim, hidden_1_out_dim, output_dim)
        self.input_dim = input_dim

        self.linear_1 = tf.keras.layers.Dense(
            units=hidden_1_out_dim,
            input_shape=(input_dim,)
        )
        self.linear_2 = tf.keras.layers.Dense(
            units=output_dim,
            input_shape=(hidden_1_out_dim,)
        )

    def call(self, input):
        output = self.linear_1(input)
        output = tf.keras.activations.tanh(output)
        output = self.linear_2(output)
        return output

    def dummy_run(self):
        """
        This method is to let python initialize the network and weights not just the computation graph.
        :return: None.
        """
        dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1, dtype=tf.float32)
        self(dummy_z)


def pmf_collection(parameter_mat):
    """
    Compute the full distribution P(X = 1, Y = 1), P(X = 1, Y = -1), P(X = -1, Y = 1) and P(X = -1, Y = -1)
    under the Ising model.
    :param parameter_mat: an n by p tensor. Each row contains a parameter for a one sample. If p = 2, we
    assume the sample correpsonds the null model. If p = 3, we assume the sample corresponds to full model.
    Columns of the parameter_mat are Jx, Jy and Jxy.
    :return: prob_mat: an n by 4 tensor. Each row contains four joint probabilities.
    """
    #############################
    # Attention !!!!!!!!!!!! There may be numerical issue. The result is slightly different with the one obtained from
    # pmf_null. Difference is 10^(-8).
    # Use pmf_null(, parameter_mat)[:, 0] * pmf_null(, parameter_mat)[:, 1] - pmf_collection(parameter_mat)[:, 0]
    # to check.
    #############################
    number_of_columns = parameter_mat.shape[1]
    if number_of_columns == 2:
        one_mat = tf.constant([
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ], dtype=tf.float32)

    elif number_of_columns == 3:
        one_mat = tf.constant([
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype=tf.float32)

    else:
        raise Exception("The shape of the parameter_mat doesn't satisfy the requirement")

    parameter_mat = tf.cast(parameter_mat, tf.float32)
    kernel_mat = tf.matmul(parameter_mat, one_mat, transpose_b=True)
    exp_kernel_mat = tf.exp(kernel_mat)
    # tf.transpose has a weird bug when running the ising_tuning script at sample size 1000.
    # prob_mat = tf.transpose(exp_kernel_mat, perm=[1, 0]) / tf.reduce_sum(exp_kernel_mat, axis=1)
    prob_mat = np.transpose(exp_kernel_mat) / tf.reduce_sum(exp_kernel_mat, axis=1)

    return np.transpose(prob_mat)

def pmf_null(x, hx):
    """
    Compute the probability P(X = x) under the null model.
    :param x: Expect either a scalar or an 1-d array of either 1 or negative 1.
    :param hx: The parameter corresponds to x.
    :return: pmf: P(X = x).
    """
    hx = tf.cast(hx, tf.float32)
    numerator = tf.exp(- x * hx)
    denominator = tf.exp(- x * hx) + tf.exp(x * hx)
    pmf = numerator / denominator

    return pmf


def log_ising_pmf(x_y_mat, parameter_mat):
    """
    Compute negative log likelihood of the Ising model under the alternative and use it as the loss function for
    model training.

    :param x_y_mat: an n by 2 tensor which stores observed x's any y's.
    :param parameter_mat: a tensor of shape [n, 3]. It should be the output of an object of class IsingNetwork.
    Columns of the matrices are Jx, Jy and Jxy respectively.

    :return: negative_log_likelihood
    """
    sample_size = tf.shape(parameter_mat)[0]
    parameter_mat = tf.cast(parameter_mat, tf.float32)
    if parameter_mat.shape[1] == 2:
        zero_tensor = tf.zeros((parameter_mat.shape[0], 1))
        parameter_mat = tf.concat(values = [parameter_mat, zero_tensor], axis = 1)


    x_times_y = x_y_mat[:, 0] * x_y_mat[:, 1]
    x_times_y = tf.reshape(x_times_y, (tf.shape(x_times_y)[0], 1))
    x_y_xy_mat = tf.concat(values=[x_y_mat, x_times_y], axis=1)
    x_y_xy_mat = tf.dtypes.cast(x_y_xy_mat, tf.float32)
    dot_product_sum = tf.reduce_sum(x_y_xy_mat * parameter_mat)

    normalizing_constant = tf.constant(0., dtype=tf.float32)
    for i in tf.range(sample_size):
        parameter_vet = parameter_mat[i, :]
        one_mat = tf.constant([
            [-1., -1., -1.],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype=tf.float32)
        exponent_vet = tf.reduce_sum(parameter_vet * one_mat, axis=1)
        log_sum_exp = tf.reduce_logsumexp(exponent_vet)
        normalizing_constant += log_sum_exp

    negative_log_likelihood = dot_product_sum + normalizing_constant
    return negative_log_likelihood


def generate_x_y_mat(sample_size, p_mat):
    """
    Generate an x_y_mat according to the p_mat.
    :param sample_size: An integer.
    :param p_mat: An sample_size x 4 matrix. The columns correspond to P(X = 1, Y = 1), P(X = 1, Y = -1),
    P(X = -1, Y = 1) and P(X = -1, Y = -1).
    :return:
    x_y_mat: x_y_mat: An n by 2 dimension numpy array / tensor. Each column contains only 1 and -1.
    """
    raw_sample_mat = np.zeros((sample_size, 4))
    for i in np.arange(sample_size):
        p_vet = p_mat[i, :]
        raw_sample = np.random.multinomial(1, p_vet)
        raw_sample_mat[i, :] = raw_sample

    conversion_mat = np.array([
        [1, 1], [1, -1], [-1, 1], [-1, -1]
    ])
    x_y_mat = raw_sample_mat.dot(conversion_mat)

    return x_y_mat


def generate_x_y_mat_ising(ising_network, z_mat, null_boolean, sample_size):
    """
    The function will generate the matrix of responses (x, y) using the Ising model.
    :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
    Usually, it is the output of the generate_z_mat method.
    :return:
    x_y_mat: An n by 2 dimension numpy array / tensor. Each column contains only 1 and -1.
    """
    true_parameter_mat = ising_network(z_mat)
    if null_boolean:
        p_equal_1_mat = pmf_null(1, true_parameter_mat)
        x_y_mat = np.random.binomial(n=1, p=p_equal_1_mat, size=(sample_size, 2)).astype(np.float32) * 2 - 1
        return x_y_mat
    else:
        p_mat = pmf_collection(true_parameter_mat)
        # Recall that the column of p_mat corresponds to P(X = 1, Y = 1), P(X = 1, Y = -1), P(X = -1, Y = 1) and
        # P(X = -1, Y = -1).
        x_y_mat = generate_x_y_mat(sample_size, p_mat)

        return x_y_mat


def mixture_model_x_y_mat(label_vet, p_x_equal_y):
    """
    The function will generate the matrix of responses (x, y) using the mixture model.
    :param label_vet: A vector of integers which should be one of the outputs of the make_blobs function.
    :param p_x_eqaul_y: A scalar which is P(X = Y).

    :return:
    x_y_mat: An n by 2 dimension numpy array / tensor. Each column contains only 1 and -1.
    p_mat: An n by 4 dimension numpy array. 4 columns are P(X = 1, Y = 1), P(X = 1, Y = -1), P(X = -1, Y = 1) and
    P(X = -1, Y = -1).
    """
    sample_size = label_vet.shape[0]
    group_0_length = np.sum(label_vet == 0)
    remaining_group_length = sample_size - group_0_length

    x_y_mat = np.zeros((sample_size, 2))
    x_y_mat[label_vet != 0, :] = 2 * np.random.binomial(n=1, p=0.5, size=(remaining_group_length, 2)) - 1

    p_equal = p_x_equal_y / 2.
    p_different = 0.5 - p_equal
    p_vet = np.array([p_equal, p_different, p_different, p_equal])

    p_mat_group_0 = np.tile(p_vet, (group_0_length, 1))
    x_y_mat[label_vet == 0, :] = generate_x_y_mat(sample_size=group_0_length, p_mat=p_mat_group_0)

    p_mat = np.repeat(0.25, sample_size * 4).reshape(sample_size, 4)
    p_mat[np.argwhere(label_vet == 0).squeeze(), :] = p_mat_group_0

    return x_y_mat, p_mat


def data_generate_network(dim_z=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim):
    """
    The function will generate an Ising Network under the alternative.
    :param dim_z: An interger. The dimension of the random_variables we condition on.
    :param hidden_1_out_dim: A integer which is the output dimension of the hidden layer.
    :return:
    null_network_generate: An instance of the IsingNetwork class.
    alt_network_generate: An instance of the IsingNetwork class.
    weights_list: A list containing weights of each layers.
    """
    # Create the alternative network.
    alt_network_generate = IsingNetwork(dim_z, hidden_1_out_dim, 3)
    alt_network_generate.dummy_run()

    linear_1_weight_array = tf.random.normal(shape=(dim_z, hidden_1_out_dim), mean=0, stddev=1)
    linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

    linear_2_weight_array = tf.random.normal(shape=(hidden_1_out_dim, 3), mean=0, stddev=1)
    linear_2_bias_array = tf.zeros(shape=(3,))

    weights_list = [linear_1_weight_array, linear_1_bias_array,
                    linear_2_weight_array, linear_2_bias_array]
    alt_network_generate.set_weights(weights_list)

    # Create the null network.
    null_network_generate = IsingNetwork(dim_z, hidden_1_out_dim, 2)
    null_network_generate.dummy_run()

    null_linear_2_weight_array = linear_2_weight_array[:, :2]
    null_linear_2_bias_array = linear_2_bias_array[: 2]

    null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                       null_linear_2_weight_array, null_linear_2_bias_array])

    return null_network_generate, alt_network_generate, weights_list


##################################
# Functions for parameter tuning #
##################################
def kl_divergence(p_mat_true, p_mat_predicted, isAverage):
    """
    Compute the KL divergence between two discrete distribution. KL(True distribution,
    Fitted distribution) (/ n). See Wikipedia for detail description.

    :param p_mat_true: An n x p numpy array / tensor. n is the sample size and p is the number of values in the support.
    This is the matrix containing the pmf of the true distribution.
    :param p_mat_predicted: An n x p numpy array / tensor. n is the sample size and p is the number of values in the
    support. This is the matrix containing the pmf of the fitted distribution.
    :param isAverage: a boolean value. If it is true, then the function will return the average kl divergence of the
    sample. Otherwise, it will return the kl divergence between distribiutions for each sample (Z)_.

    :return:
    kl_divergence_scalr: a scalar
    or kl_divergence_list: a list
    """
    assert(p_mat_true.shape == p_mat_predicted.shape)
    if len(p_mat_true.shape) == 1:
        p_mat_true = p_mat_true.reshape(1, -1)
        p_mat_predicted = p_mat_predicted.reshape(1, -1)

    kl_divergence_mat = p_mat_true * np.log2(p_mat_true / p_mat_predicted)
    if isAverage:
        kl_divergence_scalar = np.sum(kl_divergence_mat) / p_mat_true.shape[0]
        return kl_divergence_scalar
    else:
        kl_divergence_list = np.sum(kl_divergence_mat, axis=1)
        return kl_divergence_list


def kl_divergence_ising(true_parameter_mat, predicted_parameter_mat, isAverage):
    """
    Compute the average or individual KL divergence between two sets of Ising models. KL(True distribution,
    Fitted distribution) (/ n). See Wikipedia for detail description.

    :param true_parameter_mat: An n by p tensor storing parameters for the true distribution. Each row contains a
    parameter for a one sample. If p = 2, we assume the sample is under the null model. If p = 3, we assume the
    sample is under the full model.
    :param predicted_parameter_mat: an n by p tensor storing parameters for the fitted distribution. We again assume
    that p can either be 2 or 3.
    :param isAverage: a boolean value. If it is true, then the function will return the average kl divergence of the
    sample. Otherwise, it will return the kl divergence between distribiutions for each sample (Z)_.

    :return: kl_divergence_scalr: a scalar
    or kl_divergence_list: a list
    """
    p_mat_true = pmf_collection(true_parameter_mat)
    p_mat_predicted = pmf_collection(predicted_parameter_mat)
    kl_divergence_scalar = kl_divergence(p_mat_true, p_mat_predicted, isAverage)

    return kl_divergence_scalar


#########################################
# Class for the simulation and training #
#########################################
class IsingTunning:
    def __init__(self, z_mat, x_y_mat, ising_network, learning_rate=hp.learning_rate,
                 buffer_size=hp.buffer_size, batch_size=hp.batch_size, max_epoch=250):
        """
        Create a class which can generate data and train a network. It is used to get training oracle information
        such as training epoch.
        :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.

        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.batch function.
        :param epoch: A scalar indicating the number of times training process pass through the data set.
        """
        self.z_mat = z_mat
        self.x_y_mat = x_y_mat
        self.sample_size = z_mat.shape[0]
        self.ising_network = ising_network
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch

    def train_test_split(self, test_percentage):
        """
        Create and split the full data into the training data and the test data. test_percentage of the data are used
        for the test data.
        :param test_percentage: A scalar between 0 and 1.
        :return:
        train_ds: A Tensorflow dataset which is used as the training data.
        test_ds: A Tensorflow dataset which is used as the test data.
        """
        full_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, self.x_y_mat))
        full_ds = full_ds.shuffle(self.buffer_size)

        test_size = tf.cast(self.sample_size * test_percentage, tf.int32).numpy()
        test_ds = full_ds.take(test_size).batch(test_size)
        train_ds = full_ds.skip(test_size).batch(self.batch_size)

        return train_ds, test_ds

    def tuning(self, print_loss_boolean, test_percentage=0.1, true_parameter_mat=None, p_mat_true=None):
        """
        Train and tune a neural network on Ising data.

        :param true_parameter_mat: An n by p tensor storing parameters for the true distribution. Each row contains a
        parameter for a one sample. If p = 2, we assume the sample is under the null model. If p = 3, we assume the
        sample is under the full model.
        :param p_mat_true: An n x p numpy array / tensor. n is the sample size and p is the number of values in the
        support. This is the matrix containing the pmf of the true distribution.
        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.

        :param p_mat_true: An n x p numpy array / tensor. n is the sample size and p is the number of values in the
        support. This is the matrix containing the pmf of the true distribution.

        :return:
        result_dict: A dictionary which contains two keys which are "loss_array" and "ising_par".
        result_dict["loss_array"] is a 2 by epoch numpy of which the first row stores the (- 2 * LogLikelihood), the
        second row stores the (- 2 * LogLikelihood) on the test set; and the third row stores the kl divergence on the
        full data set.
        result_dict["ising_parameters"] stores a tensor which is
        the fitted value of parameters in the full Ising Model.
        """
        assert(true_parameter_mat is None or p_mat_true is None)
        assert(not(true_parameter_mat is None and p_mat_true is None))

        # Prepare training data.
        train_ds, test_ds = self.train_test_split(test_percentage = test_percentage)

        # Prepare storage for results.
        loss_kl_array = np.zeros((3, self.max_epoch))
        result_dict = dict()

        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        iteration = 0
        while iteration < self.max_epoch:
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = self.ising_network(z_batch)
                    loss = log_ising_pmf(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, self.ising_network.variables)
                optimizer.apply_gradients(grads_and_vars = zip(grads, self.ising_network.variables))
            print(f"Finished training{iteration} ")
            # for z_batch_test, x_y_batch_test in test_ds:
            #     predicted_parameter_mat_test = self.ising_network(z_batch_test)
            #     likelihood_on_test = log_ising_pmf(x_y_batch_test, predicted_parameter_mat_test)
            predicted_parameter_mat_test = self.ising_network(self.z_mat)
            likelihood_on_test = log_ising_pmf(self.x_y_mat, predicted_parameter_mat_test)

            if iteration % 10 == 0 and print_loss_boolean:
                print("Sample size %d, Epoch %d" % (self.sample_size, iteration))
                print("The loss is %f " % loss)
                print("The test loss is %f" % likelihood_on_test)

            predicted_parameter_mat = self.ising_network(self.z_mat)
            if p_mat_true is not None:
                p_mat_predicted = pmf_collection(predicted_parameter_mat)
                print(f"{self.sample_size} Finished pmf collection {iteration}")
                kl_on_full_data = kl_divergence(p_mat_true, p_mat_predicted, True)
                print(f"{self.sample_size} Finished kl {iteration}")
            else:
                kl_on_full_data = kl_divergence_ising(true_parameter_mat, predicted_parameter_mat, True)

            loss_kl_array[0, iteration] = loss.numpy()
            loss_kl_array[1, iteration] = likelihood_on_test
            loss_kl_array[2, iteration] = kl_on_full_data

            iteration += 1


        result_dict["loss_array"] = loss_kl_array
        predicted_parameter_mat = self.ising_network(self.z_mat)
        result_dict["ising_parameters"] = predicted_parameter_mat

        return result_dict


class IsingTrainingPool:
    def __init__(self, z_mat, x_y_mat, epoch, ising_network, learning_rate=hp.learning_rate,
                 buffer_size=hp.buffer_size, batch_size=hp.batch_size):
        """
        Create a class which can generate data and train a network.
        :param z_mat: An n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
        :param x_y_mat: An n by 2 dimension numpy array / tensor. Each column contains only 1 and -1.
        :param ising_network: A tf.kears.model class. It is the nerual network to be fitted.
        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.batch function.
        :param epoch: A scalar indicating the number of times training process pass through the data set.
        """
        self.z_mat = z_mat
        self.x_y_mat = x_y_mat
        self.ising_network = ising_network
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch = epoch


    def trainning(self):
        """
        Train a neural network.
        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.
        :return: result_dict: A dictionary which contains two keys which are "loss_array" and "ising_par".
        result_dict["loss_array"] is a 2 by epoch numpy of which the first row stores the (- 2 * LogLikelihood) and the
        second row stores the kl divergences. result_dict["ising_parameters"] stores a tensor which is the fitted value
        of parameters in the full Ising Model.
        """
        """
                # Prepare training data.
        train_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, x_y_mat))
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)
        """

        # Prepare training data.
        train_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, self.x_y_mat))
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = self.ising_network(z_batch)
                    loss = log_ising_pmf(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, self.ising_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, self.ising_network.variables))

        predicted_parameter_mat = self.ising_network.predict(self.z_mat)

        return predicted_parameter_mat


class WrongIsingNetwork(tf.keras.Model):
    def __init__(self, input_dim, hidden_1_out_dim, hidden_2_out_dim, output_dim):
        super().__init__(input_dim, hidden_1_out_dim, hidden_2_out_dim, output_dim)

        self.input_dim = input_dim

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
            input_shape=(hidden_2_out_dim,)
        )

    def call(self, input):
        output = self.linear_1(input)
        output = tf.keras.activations.relu(output)
        output = self.linear_2(output)
        output = tf.keras.activations.elu(output)
        output = self.linear_3(output)
        return output

    def dummy_run(self):
        """
        This method is to let python initialize the network and weights not just the computation graph.
        :return: None.
        """
        dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1, dtype=tf.float32)
        self(dummy_z)


class MixutureIsingNetwork(tf.keras.Model):
    def __init__(self, input_dim, hidden_1_out_dim, hidden_2_out_dim, output_dim):
        super().__init__(input_dim, hidden_1_out_dim, hidden_2_out_dim, output_dim)

        self.input_dim = input_dim

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
            input_shape=(hidden_2_out_dim,)
        )

    def call(self, input):
        output = self.linear_1(input)
        output = tf.keras.activations.relu(output)
        output = self.linear_2(output)
        output = tf.keras.activations.tanh(output)
        output = self.linear_3(output)
        return output

    def dummy_run(self):
        """
        This method is to let python initialize the network and weights not just the computation graph.
        :return: None.
        """
        dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1, dtype=tf.float32)
        self(dummy_z)


# class MixutureIsingNetworkOneLayer(tf.keras.Model):
#     def __init__(self, input_dim, hidden_1_out_dim, output_dim):
#         super().__init__(input_dim, hidden_1_out_dim, output_dim)
#
#         self.input_dim = input_dim
#
#         self.linear_1 = tf.keras.layers.Dense(
#             units=hidden_1_out_dim,
#             input_shape=(input_dim,)
#         )
#
#         self.linear_2 = tf.keras.layers.Dense(
#             units=output_dim,
#             input_shape=(hidden_1_out_dim,)
#         )
#
#
#     def call(self, input):
#         output = self.linear_1(input)
#         output = tf.keras.activations.tanh(output)
#         output = self.linear_2(output)
#         return output
#
#     def dummy_run(self):
#         """
#         This method is to let python initialize the network and weights not just the computation graph.
#         :return: None.
#         """
#         dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1, dtype=tf.float32)
#         self(dummy_z)





































############################################## Freeze indefinitely.
class IsingTraining_tf_function:
    # Still need debugging.
    def __init__(self, z_dataset, x_y_dataset, z_dim, hidden_1_out_dim, sample_size,
                 learning_rate, buffer_size, batch_size, epoch):
        """
        Create a class which can generate data and train a network.
        :param z_dataset:                                         . This is the data we condition on.
        :param x_y_dataset:
        :param z_dim
        :param hidden_1_out_dim: A integer which is the output dimension of the hidden layer.
        :param sample_size
        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.batch function.
        :param epoch: A scalar indicating the number of times training process pass through the data set.
        """
        self.z_dataset = z_dataset
        self.x_y_dataset = x_y_dataset
        self.z_dim = z_dim
        self.hidden_1_out_dim = hidden_1_out_dim
        self.sample_size = sample_size
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch = epoch

    def merge_z_x_y_dataset(self):
        def dt_iterator():
            for z, x_y in zip(self.z_dataset, self.x_y_dataset):
                yield (z.numpy(), x_y.numpy())

        merged_dataset = tf.data.Dataset.from_generator(dt_iterator, output_types=(tf.float32, tf.float32),
                                                        output_shapes=(tf.TensorShape(3), tf.TensorShape(2)))

        return merged_dataset

    def trainning(self):
        """
        Train a neural network.
        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.
        :return: result_dict: A dictionary which contains two keys which are "loss_array" and "ising_par".
        result_dict["loss_array"] is a 2 by epoch numpy of which the first row stores the (- 2 * LogLikelihood) and the
        second row stores the kl divergences. result_dict["ising_parameters"] stores a tensor which is the fitted value
        of parameters in the full Ising Model.
        """
        """
                # Prepare training data.
        train_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, x_y_mat))
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)
        """

        train_ds = self.merge_z_x_y_dataset()
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)

        train_network = IsingNetwork(self.z_dim, self.hidden_1_out_dim, 3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        batch_predicted_parameter_mat = tf.Variable(tf.zeros((self.batch_size, 3), dtype=tf.float32), trainable=True)
        loss = tf.Variable(0, dtype=tf.float32, trainable=True)
        predicted_parameter_mat = tf.Variable(tf.zeros((self.sample_size, 3), dtype=tf.float32))

        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat.assign(train_network(z_batch))
                    loss.assign(log_ising_pmf(x_y_batch, batch_predicted_parameter_mat))
                grads = tape.gradient(loss, train_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

        """
        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = train_network(z_batch)
                    loss = log_ising_pmf(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, train_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

        predicted_parameter_mat = tf.Variable( tf.zeros( (self.sample_size, 3), dtype = tf.float32) )
        """

        for z_mat in self.z_dataset.batch(self.sample_size):
            predicted_parameter_mat.assign(train_network(z_mat))

        # predicted_parameter_mat = train_network.predict(self.z_dataset.batch(self.batch_size))

        return predicted_parameter_mat


##################################################################
# Helper functions to load txt data file into Tensorflow dataset #
##################################################################
# Not in use!

# z_dataset = tf.data.TextLineDataset("./data/z_mat/z_mat_30.txt")
# x_y_dataset = tf.data.TextLineDataset("./data/null/x_y_mat_30_0.txt")

# Helper functions to read data using tensorflow dataset module.
def parse_fnc(line):
    string_vals = tf.strings.split([line]).values
    return tf.strings.to_number(string_vals, tf.float32)


# z_dataset = z_dataset.map(parse_fnc)
# x_y_dataset = x_y_dataset.map(parse_fnc)

def tf_load_z_dataset(sample_size_tensor):
    z_file_format = tf.strings.format("./data/z_mat/z_mat_{}.txt", inputs=sample_size_tensor)
    z_dataset = tf.data.TextLineDataset(z_file_format)
    z_dataset = z_dataset.map(parse_fnc)

    return z_dataset


# z_dataset = tf_load_z_dataset(tf.constant(30))
# print(list(z_mat.take(1)))

def tf_load_x_y_dataset_null(sample_size_tensor, simulation_times_tensor):
    x_y_file_format = tf.strings.format("./data/null/x_y_mat_{}_{}.txt", inputs=(sample_size_tensor,
                                                                                 simulation_times_tensor))
    x_y_dataset = tf.data.TextLineDataset(x_y_file_format)
    x_y_dataset = x_y_dataset.map(parse_fnc)

    return x_y_dataset


# x_y_dataset = tf_load_x_y_dataset_null(tf.constant(30), tf.constant(1))
# print(list(x_y_dataset.take(1)))

def tf_load_x_y_dataset_alt(sample_size_tensor, simulation_times_tensor):
    x_y_file_format = tf.strings.format("./data/alt/x_y_mat_{}_{}.txt", inputs=(sample_size_tensor,
                                                                                simulation_times_tensor))
    x_y_dataset = tf.data.TextLineDataset(x_y_file_format)
    x_y_dataset = x_y_dataset.map(parse_fnc)

    return x_y_dataset

# x_y_dataset = tf_load_x_y_dataset_alt(tf.constant(30), tf.constant(1))
# print(list(x_y_dataset.take(1)))







######################### Recycle
"""


class IsingTrainingPool:
    def __init__(self, z_mat, epoch, hidden_1_out_dim=hp.hidden_1_out_dim, learning_rate=hp.learning_rate,
                 buffer_size=hp.buffer_size, batch_size=hp.batch_size):
        
        Create a class which can generate data and train a network.
        :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
        :param hidden_1_out_dim: A scalar which is the output dimension of the hidden layer.
        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.batch function.
        :param epoch: A scalar indicating the number of times training process pass through the data set.
        
        self.z_mat = z_mat
        self.hidden_1_out_dim = hidden_1_out_dim
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch = epoch

        self.sample_size = z_mat.shape[0]
        self.dim_z = z_mat.shape[1]

    def trainning(self, x_y_mat):
        
        Train a neural network.
        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.
        :return: result_dict: A dictionary which contains two keys which are "loss_array" and "ising_par".
        result_dict["loss_array"] is a 2 by epoch numpy of which the first row stores the (- 2 * LogLikelihood) and the
        second row stores the kl divergences. result_dict["ising_parameters"] stores a tensor which is the fitted value
        of parameters in the full Ising Model.
        
        
        # Prepare training data.
        train_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, x_y_mat))
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)
        

        # Prepare training data.
        train_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, x_y_mat))
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)

        train_network = IsingNetwork(self.dim_z, self.hidden_1_out_dim, 3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = train_network(z_batch)
                    loss = log_ising_pmf(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, train_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

        predicted_parameter_mat = train_network.predict(self.z_mat)

        return predicted_parameter_mat

"""