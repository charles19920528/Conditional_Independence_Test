import tensorflow as tf
# Numpy is for the generate_x_y_mat_ising method and trainning method in the IsingSimulation class.
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
        assume the sample corresponding to the null model. If p = 3, we assume the sample corresponds to full model.
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
    # tf.transpose has a weird bug when running the ising_tuning script at sample size 1000.
    # prob_mat = tf.transpose(exp_kernel_mat, perm=[1, 0]) / tf.reduce_sum(exp_kernel_mat, axis=1)
    prob_mat = np.transpose(exp_kernel_mat) / tf.reduce_sum(exp_kernel_mat, axis=1)
    prob_mat = np.transpose(prob_mat)

    return prob_mat


"""
def pmf_null(x, hx):
    ##################
    # To be deleted and be replaced by the pmf_collection
    ###################
    hx = tf.cast(hx, tf.float32)
    numerator = tf.exp(- x * hx)
    denominator = tf.exp(- x * hx) + tf.exp(x * hx)
    pmf = numerator / denominator

    return pmf
"""


def log_ising_likelihood(x_y_mat, parameter_mat):
    """
    Compute negative log likelihood of the Ising model. The function can be used as the loss function for
    model training.

    :param x_y_mat: an n by 2 tensor which stores observed x's any y's.
    :param parameter_mat: a tensor of shape [n, 2] or [n, 3]. It should be the output of an object of class
        IsingNetwork. Columns of the matrices are Jx, Jy and Jxy respectively.

    :return: negative_log_likelihood
    """
    sample_size = tf.shape(parameter_mat)[0]
    parameter_mat = tf.cast(parameter_mat, tf.float32)
    if parameter_mat.shape[1] == 2:
        zero_tensor = tf.zeros((parameter_mat.shape[0], 1))
        parameter_mat = tf.concat(values=[parameter_mat, zero_tensor], axis=1)

    x_times_y = x_y_mat[:, 0] * x_y_mat[:, 1]
    x_times_y = tf.reshape(x_times_y, (-1, 1))
    x_y_xy_mat = tf.concat(values=[x_y_mat, x_times_y], axis=1)
    x_y_xy_mat = tf.dtypes.cast(x_y_xy_mat, tf.float32)
    dot_product_sum = tf.reduce_sum(x_y_xy_mat * parameter_mat)

    normalizing_constant = tf.constant(0., dtype=tf.float32)
    one_mat = tf.constant([
        [-1., -1., -1.],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
    ], dtype=tf.float32)
    for i in tf.range(sample_size):
        parameter_vet = parameter_mat[i, :]
        exponent_vet = tf.reduce_sum(parameter_vet * one_mat, axis=1)
        log_sum_exp = tf.reduce_logsumexp(exponent_vet)
        normalizing_constant += log_sum_exp

    negative_log_likelihood = dot_product_sum + normalizing_constant
    return negative_log_likelihood


def generate_x_y_mat(p_mat):
    """
    Generate an x_y_mat according to the p_mat. The ith row of x_y_mat corresponds to a sample (x_i, y_i) following
    the distribution specified in the ith row p_mat.

    :param p_mat: An sample_size x 4 matrix. The columns correspond to P(X = 1, Y = 1), P(X = 1, Y = -1),
        P(X = -1, Y = 1) and P(X = -1, Y = -1).

    :return:
        x_y_mat: An sample_size by 2 dimension matrix. Each column contains only 1 and -1.
    """
    sample_size = p_mat.shape[0]
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


def generate_x_y_mat_ising(ising_network, z_mat):
    """
    The function will generate the x_y_mat using the Ising model.

    :param ising_network: An instance of IsingNetwork class.
    :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
        Usually, it is the output of the generate_z_mat method.

    :return:
        x_y_mat: An n by 2-dimension matrix. Each column contains only 1 and -1. The first column corresponds to X.
    """
    true_parameter_mat = ising_network(z_mat)
    p_mat = pmf_collection(true_parameter_mat)
    x_y_mat = generate_x_y_mat(p_mat)

    return x_y_mat


def data_generate_network(dim_z=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim):
    """
    The function will generate a pair of IsingNetwork instances under the null and the alternative. The null network
    has one less set of weights which corresponds to Jxy. Every other weights are exactly the same. Weights are
    generated using a standard normal distribution. This function is  just a wrap-up function and the only purpose is to
    make the code cleaner.

    :param dim_z: An interger. The dimension of the random_variables we condition on.
    :param hidden_1_out_dim: A integer which is the output dimension of the hidden layer.

    :return: 1.null_network_generate: An instance of the IsingNetwork class. 2.alt_network_generate: An instance of the
            IsingNetwork class. 3.weights_list: A list containing weights of each layers.
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


########################
# pmf for mixture data #
########################
def conditional_pmf_collection_mixture(z_mat, is_null_boolean, cut_off_radius):
    """
    Compute the conditoinal pmf for the mixture data.

    :param z_mat: An n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
    :param is_null_boolean: A boolean value to indicate if we compute the pmf under the independence assumption (H0).
    :param cut_off_radius: A positive scalar which we use to divide sample into two groups based on the norm of z.

    :return: p_mat: An n by 4 dimension numpy array. 4 columns are P(X = 1, Y = 1), P(X = 1, Y = -1), P(X = -1, Y = 1)
        and P(X = -1, Y = -1).
    """
    less_than_cut_off_boolean = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=z_mat) < cut_off_radius
    nrow = z_mat.shape[0]
    if is_null_boolean:
        p_mat = np.repeat(0.25, nrow * 4).reshape(nrow, 4)
        helper_pmf_vet = np.array([0, 0, 0, 1]).reshape(1, 4)
        p_mat[~less_than_cut_off_boolean] = np.tile(helper_pmf_vet, (sum(~less_than_cut_off_boolean), 1))

        return p_mat

    else:
        p_mat = np.tile([0, 0.5, 0.5, 0], nrow).reshape(nrow, 4)
        helper_pmf_vet = np.array([0.5, 0, 0, 0.5])
        p_mat[~less_than_cut_off_boolean] = np.tile(helper_pmf_vet, (sum(~less_than_cut_off_boolean), 1))

        return p_mat


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
    assert p_mat_true.shape == p_mat_predicted.shape

    none_zero_mass_boolean = p_mat_true != 0

    kl_divergence_mat = np.zeros(p_mat_true.shape)
    kl_divergence_mat[none_zero_mass_boolean] = p_mat_true[none_zero_mass_boolean] * \
                                                np.log(p_mat_true[none_zero_mass_boolean] /
                                                       p_mat_predicted[none_zero_mass_boolean])
    np.nan_to_num(x=kl_divergence_mat, copy=False, nan=2)
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
class IsingTrainingTuning:
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


    def train_test_split(self, number_of_test_samples):
        """
        Create and split the full data into the training data and the test data. test_percentage of the data are used
        for the test data.

        :param number_of_test_samples: An integer which is the number of data used as validation set.

        :return:
        train_array_tuple: A tuple of length 2 containing z_mat and x_y_mat for the training data
        test_array_tuple: A tuple of length 2 containing z_mat and x_y_mat for the test data.
        """
 #       test_size = tf.cast(self.sample_size * test_percentage, tf.int8).numpy()
        indices_vet = np.random.permutation(self.sample_size)
        training_indices_vet, test_indices_vet = indices_vet[number_of_test_samples:], \
                                                 indices_vet[:number_of_test_samples]

        train_array_tuple = (self.z_mat[training_indices_vet, :], self.x_y_mat[training_indices_vet, :])
        test_array_tuple = (self.z_mat[test_indices_vet, :], self.x_y_mat[test_indices_vet, :])

        # full_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, self.x_y_mat))
        # full_ds = full_ds.shuffle(self.buffer_size)
        #
        # test_size = tf.cast(self.sample_size * test_percentage, tf.int32).numpy()
        # test_ds = full_ds.take(test_size).batch(test_size)
        # train_ds = full_ds.skip(test_size).batch(self.batch_size)

        return train_array_tuple, test_array_tuple


    def tuning(self, print_loss_boolean, is_null_boolean, number_of_test_samples, cut_off_radius=None,
               true_weight_array=None):
        """
        Train and tune a neural network on Ising data.

        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.
        :param test_percentage: A scalar between 0 and 1.
        :param is_null_boolean: A boolean value to indicate if we compute the pmf under the independence assumption
        (H0).

        :param true_parameter_mat: An n by p tensor storing parameters for the true distribution. Each row contains a
        parameter for a one sample. If p = 2, we assume the sample is under the null model. If p = 3, we assume the
        sample is under the full model.
        :param p_mat_true: An n x p numpy array / tensor. n is the sample size and p is the number of values in the
        support. This is the matrix containing the pmf of the true distribution.


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
        assert cut_off_radius is None or true_weight_array is None, \
            "Both cut_off_radius and true_weight_array are supplied."
        assert cut_off_radius is not None or true_weight_array is not None, \
            "Neither cut_off_radius nor true_weight_array are supplied."

        # Prepare training and test data.
        train_array_tuple, test_array_tuple = self.train_test_split(number_of_test_samples=number_of_test_samples)
        train_ds = tf.data.Dataset.from_tensor_slices(train_array_tuple)
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)
        test_z_mat, test_x_y_mat = test_array_tuple

        # Compute true pmf on the test data.
        if true_weight_array is None:
            true_test_p_mat = conditional_pmf_collection_mixture(z_mat=test_z_mat, is_null_boolean=is_null_boolean,
                                                                 cut_off_radius=cut_off_radius)
        else:
            true_network = IsingNetwork(hp.dim_z, hp.hidden_1_out_dim, 3)
            true_network.dummy_run()
            true_network.set_weights(true_weight_array)

            true_test_parameter_mat = true_network(test_z_mat)
            if is_null_boolean:
                true_test_parameter_mat = true_test_parameter_mat[:, :2]

            true_test_p_mat = pmf_collection(parameter_mat=true_test_parameter_mat)

        # Prepare storage for results.
        loss_kl_array = np.zeros((3, self.max_epoch))
        result_dict = dict()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        iteration = 0
        while iteration < self.max_epoch:

            # print(f"start training epoch {iteration}")
            # Training loop.
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = self.ising_network(z_batch)
                    loss = log_ising_likelihood(x_y_batch, batch_predicted_parameter_mat)

                    # print(f"finish compute log pmf {iteration}")

                grads = tape.gradient(loss, self.ising_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, self.ising_network.variables))

            # print(f"Finished training{iteration} ")

            # Compute likelihood and  kl on test data  test_z_mat, test_x_y_mat
            predicted_test_parameter_mat = self.ising_network(test_z_mat)
            likelihood_on_test = log_ising_likelihood(test_x_y_mat, predicted_test_parameter_mat)
            predicted_test_p_mat = pmf_collection(parameter_mat=predicted_test_parameter_mat)
            kl_on_test_data = kl_divergence(p_mat_true=true_test_p_mat, p_mat_predicted=predicted_test_p_mat,
                                            isAverage=True)

            # print(f"{self.sample_size} Finished kl {iteration}")

            if iteration % 10 == 0 and print_loss_boolean:
                print("Sample size %d, Epoch %d" % (self.sample_size, iteration))
                print("The training loss is %f " % loss)
                print("The test loss is %f" % likelihood_on_test)
                print("The kl on the test data is %f" % kl_on_test_data)

            loss_kl_array[0, iteration] = loss.numpy()
            loss_kl_array[1, iteration] = likelihood_on_test
            loss_kl_array[2, iteration] = kl_on_test_data

            iteration += 1

        result_dict["loss_array"] = loss_kl_array
        predicted_parameter_mat = self.ising_network(self.z_mat)
        result_dict["ising_parameters"] = predicted_parameter_mat

        return result_dict


    def train_compute_test_statistic(self, print_loss_boolean, number_of_test_samples):
        """

        :param print_loss_boolean:
        :param number_of_test_samples:
        :return:
        """
        # Prepare training and test data.
        train_array_tuple, test_array_tuple = self.train_test_split(number_of_test_samples=number_of_test_samples)
        train_ds = tf.data.Dataset.from_tensor_slices(train_array_tuple)
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)
        test_z_mat, _ = test_array_tuple

        result_dict = dict()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        iteration = 0
        while iteration < self.max_epoch:
            # Training loop.
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = self.ising_network(z_batch)
                    loss = log_ising_likelihood(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, self.ising_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, self.ising_network.variables))

            if iteration % 10 == 0 and print_loss_boolean:
                print("Sample size %d, Epoch %d" % (self.sample_size, iteration))
                print("The training loss is %f " % loss)

            iteration += 1

        predicted_test_parameter_mat = self.ising_network(test_z_mat)
        result_dict["predicted_parameter_mat"] = predicted_test_parameter_mat

        jxy_squared_vet = np.square(predicted_test_parameter_mat[:, 2])
        jxy_squared_mean = np.mean(jxy_squared_vet)
        result_dict["test_statistic"] = jxy_squared_mean

        return result_dict


class ForwardEluLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim):
        super(ForwardEluLayer, self).__init__()

        self.linear = tf.keras.layers.Dense(
            units=hidden_dim,
            input_shape=(input_dim,)
        )
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        output = self.linear(inputs)
        output = tf.keras.activations.elu(output)

        return output


class FullyConnectedNetwork(tf.keras.Model):
    def __init__(self, number_forward_elu_layers, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()

        self.input_dim = input_dim
        self.number_forward_elu_layers = number_forward_elu_layers

        self.initial_block = ForwardEluLayer(input_dim=input_dim, hidden_dim=hidden_dim)

        if number_forward_elu_layers > 1:
            self.feed_forward_rest_vet = [ForwardEluLayer(input_dim=hidden_dim, hidden_dim=hidden_dim) for _ in
                                          np.arange(number_forward_elu_layers - 1)]

        self.final_linear = tf.keras.layers.Dense(
            units=output_dim,
            input_shape=(hidden_dim,)
        )

    def call(self, inputs):
        output = self.initial_block(inputs)
        if self.number_forward_elu_layers == 1:
            output = self.final_linear(output)
        else:
            for i in np.arange(self.number_forward_elu_layers - 1):
                output = self.feed_forward_rest_vet[i](output)
            output = self.final_linear(output)

        return output

    def dummy_run(self):
        """
        This method is to let python initialize the network and weights not just the computation graph.
        :return: None.
        """
        dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1, dtype=tf.float32)
        self(dummy_z)


