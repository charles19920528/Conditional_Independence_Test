import tensorflow as tf
import numpy as np
from functools import partial
import sys
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
        This method is to initialize the network and weights not just the computation graph.
        :return: None.
        """
        dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1, dtype=tf.float32)
        self(dummy_z)


def pmf_collection(parameter_mat):
    """
    Compute the full distribution P(X = 1, Y = 1), P(X = 1, Y = -1), P(X = -1, Y = 1) and P(X = -1, Y = -1)
    under the Ising model.

    :param parameter_mat: An n by p tensor. Each row contains a parameter for a one sample. If p = 2, we
        assume the sample corresponding to the null model. If p = 3, we assume the sample corresponds to full model.
        Columns of the parameter_mat are Jx, Jy and Jxy.

    :return: prob_mat: An n by 4 tensor. Each row contains four joint probabilities.
    """
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
    prob_mat = tf.nn.softmax(kernel_mat, axis=1)

    return prob_mat


def log_ising_likelihood(x_y_mat, parameter_mat):
    """
    Compute negative log likelihood of the Ising model. The function can be used as the loss function for
    model training.

    :param x_y_mat: An n by 2 tensor which stores observed x's any y's.
    :param parameter_mat: A tensor of shape [n, 2] or [n, 3]. It should be the output of an object of class
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
    p_mat = tf.cast(p_mat, dtype=tf.float64)
    sample_size = p_mat.shape[0]
    raw_sample_mat = np.zeros((sample_size, 4))
    for i in np.arange(sample_size):
#        p_vet = p_mat[i, :]
        p_vet = p_mat[i, :] / np.sum(p_mat[i, :])
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
    :param z_mat: A n by p dimension numpy array or tensor. n is the sample size. This is the data we condition on.
        Usually, it is the output of the generate_z_mat method.

    :return:
        x_y_mat: An n by 2-dimension matrix. Each column contains only 1 and -1. The first column corresponds to X.
    """
    true_parameter_mat = ising_network(z_mat)
    p_mat = pmf_collection(true_parameter_mat)
    x_y_mat = generate_x_y_mat(p_mat)

    return x_y_mat


def data_generate_network(weights_distribution_string, dim_z=hp.dim_z, hidden_1_out_dim=hp.hidden_1_out_dim):
    """
    The function will generate a pair of IsingNetwork instances under the null and the alternative. The null network
    has one less set of weights which corresponds to Jxy. Every other weights are exactly the same. Weights are
    generated using either a standard normal or the Glorot normal distribution . This function is just a wrap-up
    function and the only purpose is to make the code cleaner.

    :param weights_distribution: A string which should either be "normal" or "glorot_normal"
    :param dim_z: An interger. The dimension of the random_variables we condition on.
    :param hidden_1_out_dim: A integer which is the output dimension of the hidden layer.

    :return: 1.null_network_generate: An instance of the IsingNetwork class. 2.alt_network_generate: An instance of the
            IsingNetwork class. 3.weights_list: A list containing weights of each layers.
    """
    if weights_distribution_string == "normal":
        weights_initializer = tf.random_normal_initializer(mean=0, stddev=1)
    elif weights_distribution_string == "glorot_normal":
        weights_initializer = tf.keras.initializers.GlorotNormal()

    # Create the alternative network.
    alt_network_generate = IsingNetwork(dim_z, hidden_1_out_dim, 3)
    alt_network_generate.dummy_run()

    linear_1_weight_array = weights_initializer(shape=(dim_z, hidden_1_out_dim))
    linear_1_bias_array = tf.zeros(shape=(hidden_1_out_dim,))

    linear_2_weight_array = weights_initializer(shape=(hidden_1_out_dim, 3))
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
    Compute the conditoinal pmf for the mixture data. Under the null, if norm(z) is less than the cut_off_radius,
    X and Y are independent Bernoulli random variables, otherwise, X = Y. Under the alternative, X is Bernoulli0.5),
    if norm(z) is less than the cut_off_radius, y = -x. Otherwise, Y = X.

    :param z_mat: An n by p dimension numpy array or tensor. n is the sample size. This is the data we condition on.
    :param is_null_boolean: A boolean value to indicate if we compute the pmf under the independence assumption (H0).
    :param cut_off_radius: A positive scalar which we use to divide sample into two groups based on the norm of z.

    :return: p_mat: An n by 4 dimension numpy array. 4 columns are P(X = 1, Y = 1), P(X = 1, Y = -1), P(X = -1, Y = 1)
        and P(X = -1, Y = -1).
    """
    less_than_cut_off_boolean = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=z_mat) < cut_off_radius
    sample_size = z_mat.shape[0]
    if is_null_boolean:
        p_mat = np.repeat(0.25, sample_size * 4).reshape(sample_size, 4)
        helper_pmf_vet = np.array([0.5, 0, 0, 0.5]).reshape(1, 4)
        p_mat[~less_than_cut_off_boolean] = np.tile(helper_pmf_vet, (sum(~less_than_cut_off_boolean), 1))

        return p_mat
    else:
        p_mat = np.tile([0, 0.5, 0.5, 0], (sample_size, 1))
        helper_pmf_vet = np.array([0.5, 0, 0, 0.5])
        p_mat[~less_than_cut_off_boolean] = np.tile(helper_pmf_vet, (sum(~less_than_cut_off_boolean), 1))

        return p_mat


##################################
# Functions for parameter tuning #
##################################
def kl_divergence_ising(true_parameter_mat, predicted_parameter_mat, isAverage):
    """
    Compute the average or individual KL divergence between two sets of Ising models. KL(True distribution,
    Fitted distribution) (/ n). See Wikipedia for detail description.

    :param true_parameter_mat: An n by p tensor storing parameters for the true distribution. Each row contains a
    parameter for a one sample. If p = 2, we assume the sample is under the null model. If p = 3, we assume the
    sample is under the full model.
    :param predicted_parameter_mat: An n by p tensor storing parameters for the fitted distribution. We again assume
    that p can either be 2 or 3.
    :param isAverage: A boolean value. If it is true, then the function will return the average kl divergence of the
    sample. Otherwise, it will return the kl divergence between distributions for each sample.

    :return: If isAverage is true, return a length n array of which each entry is the kl divergence of the sample. If
        isAverage is false, it returns the average KL divergence.
    """
    p_mat_true = pmf_collection(true_parameter_mat)
    p_mat_predicted = pmf_collection(predicted_parameter_mat)
    if isAverage:
        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    else:
        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

    return kl(p_mat_true, p_mat_predicted).numpy()


#########################################
# Class for the simulation and training #
#########################################
def train_network(train_ds, optimizer, network_model):
    """
    This private method is a helper function used in training neural network. It trains the ising_network 1 epoch on
    the train_ds

    :param train_ds: A tensorflow dataset object. This is the training data.
    :param optimizer: A tf.keras.optimizers instance.
    :param network_model: An instance of self.network_model_class.

    :return:
        A scalar which is the loss on that last batch of the training data.
    """
    for z_batch, x_y_batch in train_ds:
        with tf.GradientTape() as tape:
            batch_predicted_parameter_mat = network_model(z_batch)
            loss = log_ising_likelihood(x_y_batch, batch_predicted_parameter_mat)
        grads = tape.gradient(loss, network_model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, network_model.variables))

    return loss.numpy()

class NetworkTrainingTuning:
    def __init__(self, z_mat, x_y_mat, network_model_class, network_model_class_kwargs, epoch,
                 learning_rate=hp.learning_rate, buffer_size=hp.buffer_size, batch_size=hp.batch_size):
        """
        Create a class which can be used to get optimal oracle training information such as training epoch.

        :param z_mat: An n by p dimension numpy array or tensor. n is the sample size and p is the dimension.
            This is the data we condition on.
        :param x_y_mat: an n by p dimension numpy array or tensor. n is the sample size and p is the dimension.
            This the response.
        :param network_model_class: A subclass of tf.keras.Model with output dimension 3. An instance of the class is the
        neural network to fit on the data.
    :param network_model_class_kwargs: Keyword arguments to be passed in to the constructor of the network_model_class.
        :param epoch: An integer indicating the number of times training process pass through the data set.
        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: An integer which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: An integer which is a (hyper)parameter in the tf.data.Dataset.batch function.
        """
        self.z_mat = z_mat
        self.x_y_mat = x_y_mat
        self.sample_size = z_mat.shape[0]
        self.network_model_class = network_model_class
        self.network_model_class_kwargs = network_model_class_kwargs
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_epoch = epoch

        # self.train_indices_vet = None
        # self.test_indices_vet = None
        # self.fitted_train_p_mat = None
        # self.test_statistic = None
        # self.result_dict = None

    def train_test_split(self, number_of_test_samples):
        """
        Create and split the full data into the training data and the test data. number_of_test_samples samples are used
        as the test data.

        :param number_of_test_samples: An integer which is the number of samples used as validation set.

        :return:
        train_array_tuple: A tuple of length 2 containing z_mat and x_y_mat for the training data
            test_array_tuple: A tuple of length 2 containing z_mat and x_y_mat for the test data.
        """
        indices_vet = np.random.permutation(self.sample_size)
        train_indices_vet, test_indices_vet = indices_vet[number_of_test_samples:], indices_vet[:number_of_test_samples]

        train_array_tuple = (self.z_mat[train_indices_vet, :], self.x_y_mat[train_indices_vet, :])
        test_array_tuple = (self.z_mat[test_indices_vet, :], self.x_y_mat[test_indices_vet, :])

        return train_array_tuple, test_array_tuple, train_indices_vet, test_indices_vet


    def tuning(self, print_loss_boolean, is_null_boolean, number_of_test_samples, cut_off_radius=None,
               true_weights_array=None):
        """
        The function is used for (hyper)parameter tuning of neural network based on the data generated either by the
        Ising model or the mixture model. Either cut_off_radius or the true_weights_array should be supplied. When the
        former is supplied, we assume that the data is generated under the  mixture model. When later is provided, we
        assume the data is generated under the Ising model.

        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.
        :param number_of_test_samples: An integer which is the number of samples used as validation set.
        :param is_null_boolean: A boolean value to indicate if we compute the data is generated under the conditional
            independence assumption (H0).
        :param cut_off_radius: If supplied, it should be a scalar which is the cut_off_radius used when generating
            the mixture data.
        :param true_weights_array: If supplied , it should be an array which is the true weights of the data generating
            Ising network.

        :return:
        loss_kl_array: A 3 by self.max_epoch numpy array of which the first row stores the (- 2 * LogLikelihood),
            the second row stores the (- 2 * LogLikelihood) on the test set;
            and the third row stores the kl divergence on the test data.
        """
        assert cut_off_radius is None or true_weights_array is None, \
            "Both cut_off_radius and true_weights_array are supplied."
        assert cut_off_radius is not None or true_weights_array is not None, \
            "Neither cut_off_radius nor true_weights_array are supplied."

        # Prepare training and test data.
        train_array_tuple, test_array_tuple, _ , _= self.train_test_split(number_of_test_samples=number_of_test_samples)
        train_ds = tf.data.Dataset.from_tensor_slices(train_array_tuple)
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)
        test_z_mat, test_x_y_mat = test_array_tuple

        # Compute true pmf on the test data.
        if true_weights_array is None:
            true_test_p_mat = conditional_pmf_collection_mixture(z_mat=test_z_mat, is_null_boolean=is_null_boolean,
                                                                 cut_off_radius=cut_off_radius)
        else:
            true_network = IsingNetwork(hp.dim_z, hp.hidden_1_out_dim, 3)
            true_network.dummy_run()
            true_network.set_weights(true_weights_array)

            true_test_parameter_mat = true_network(test_z_mat)
            if is_null_boolean:
                true_test_parameter_mat = true_test_parameter_mat[:, :2]

            true_test_p_mat = pmf_collection(parameter_mat=true_test_parameter_mat)

        # Prepare for the training.
        network_model = self.network_model_class(**self.network_model_class_kwargs)

        loss_kl_array = np.zeros((3, self.max_epoch))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        epoch = 0
        while epoch < self.max_epoch:
            loss_on_the_last_batch = train_network(train_ds=train_ds, optimizer=optimizer,
                                                   network_model=network_model)

            # Compute likelihood and kl on test data.
            predicted_test_parameter_mat = network_model(test_z_mat)
            likelihood_on_test = log_ising_likelihood(test_x_y_mat, predicted_test_parameter_mat)
            predicted_test_p_mat = pmf_collection(parameter_mat=predicted_test_parameter_mat)
            kl_on_test_data = kl(true_test_p_mat, predicted_test_p_mat).numpy()

            if epoch % 10 == 0 and print_loss_boolean:
                print("Sample size %d, Epoch %d." % (self.sample_size, epoch))
                print(f"The training loss is {loss_on_the_last_batch}.")
                print("The test loss is %f." % likelihood_on_test)
                print("The kl on the test data is %f." % kl_on_test_data)

            loss_kl_array[0, epoch] = loss_on_the_last_batch
            loss_kl_array[1, epoch] = likelihood_on_test
            loss_kl_array[2, epoch] = kl_on_test_data

            epoch += 1

        return loss_kl_array


    def train_compute_test_statistic(self, print_loss_boolean, number_of_test_samples):
        """
        Train the ising_network on the data in the instance and compute a test statistic based on the partial data
        (test data).

        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.
        :param number_of_test_samples: An integer which is the number of samples used as validation set.

        :return:
            result_dict: A dictionary. result_dict["test_statistic] is a scalar which is the test statistic computed on
            partial data(test data).
            result_dict["test_indices_vet"] is an array containing indices of samples which are used as validation set.
        """
        # Prepare training and test data.
        train_array_tuple, test_array_tuple, train_indices_vet, test_indices_vet = \
            self.train_test_split(number_of_test_samples=number_of_test_samples)
        train_ds = tf.data.Dataset.from_tensor_slices(train_array_tuple)
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)
        test_z_mat, _ = test_array_tuple

        network_model = self.network_model_class(**self.network_model_class_kwargs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        epoch = 0
        while epoch < self.max_epoch:
            # Training loop.
            loss_on_the_last_batch = train_network(train_ds=train_ds, optimizer=optimizer, network_model=network_model)
            if epoch % 10 == 0 and print_loss_boolean:
                print("Sample size %d, Epoch %d." % (self.sample_size, epoch))
                print(f"The training loss is {loss_on_the_last_batch}.")

            epoch += 1

        predicted_parameter_mat = network_model(self.z_mat)
        jxy_squared_vet = np.square(predicted_parameter_mat[:, 2])

        train_jxy_squared_mean = np.mean(jxy_squared_vet[train_indices_vet])
        test_jxy_squared_mean = np.mean(jxy_squared_vet[test_indices_vet])

        jx_jy_train_mat = tf.gather(predicted_parameter_mat, train_indices_vet, axis=0)
        jx_jy_train_mat = tf.gather(jx_jy_train_mat, [0, 1], axis=1)
        fitted_train_p_mat = pmf_collection(jx_jy_train_mat)

        result_dict = {"test_test_statistic": test_jxy_squared_mean, "train_indices_vet": train_indices_vet,
                       "test_indices_vet": test_indices_vet,
                       "fitted_train_p_mat": fitted_train_p_mat, "train_test_statistic": train_jxy_squared_mean}

        return result_dict




# def __ising_bootstrap_one_trial(_, fitted_train_p_mat, z_mat, train_indices_vet, ,
#                                 network_model_class, network_model_class_kwargs, buffer_size, batch_size, learning_rate,
#                                 max_epoch):
#     """
#     Helper function to generate a boostrap sample using neural ising model and obtain bootstrap test statistic. We wiil
#     pass the function into a Pool instance.
#
#     :param _:
#     :param fitted_train_p_mat:
#     :param z_mat:
#     :param train_indices_vet:
#     :param buffer_size:
#     :param batch_size:
#     :param learning_rate:
#     :return:
#     """
#     # Resample
#     new_train_x_y_mat = generate_x_y_mat(fitted_train_p_mat)
#     train_ds = tf.data.Dataset.from_tensor_slices((z_mat[train_indices_vet, :], new_train_x_y_mat))
#     train_ds = train_ds.shuffle(buffer_size).batch(batch_size)
#
#     # Train the network.
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     network_model = network_model_class(**network_model_class_kwargs)
#     epoch = 0
#     while epoch < max_epoch:
#         _ = self.__train_network(train_ds=train_ds, optimizer=optimizer, network_model=network_model)
#         epoch += 1
#
#     predicted_test_parameter_mat = network_model(z_mat[test_indices_vet, :])
#     jxy_squared_vet = np.square(predicted_test_parameter_mat[:, 2])
#     jxy_squared_mean = np.mean(jxy_squared_vet)
#     print("One trial finished.")
#     return jxy_squared_mean
#
#     def bootstrap(self, pool, number_of_bootstrap_samples):
#         if self.result_dict is None:
#             sys.exit("train_compute_test_statistic method needs to be called before running bootstrap method")
#
#         print("Boostrap begins.")
#         bootstrap_test_statistic_vet = pool.map(self.bootstrap_one_trial, np.arange(number_of_bootstrap_samples))
#         print("Bootstrap finished.")
#
#         self.result_dict["bootstrap_test_statistic_vet"] = bootstrap_test_statistic_vet
#         p_value = sum(bootstrap_test_statistic_vet > self.test_statistic) / number_of_bootstrap_samples
#         self.result_dict["p_value"] = p_value


#####################################################
# Class for n layers fully connected neural network #
#####################################################
# This the neural network model we are going to use to fit on the data.
class ForwardLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim):
        super(ForwardLayer, self).__init__()

        self.linear = tf.keras.layers.Dense(
            units=hidden_dim,
            input_shape=(input_dim,)
        )
#        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        output = self.linear(inputs)
        output = tf.keras.activations.elu(output)

        return output


class FullyConnectedNetwork(tf.keras.Model):
    # This is the class network we fit on the data.
    def __init__(self, number_forward_layers, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()

        self.input_dim = input_dim
        self.number_forward_elu_layers = number_forward_layers

        self.initial_block = ForwardLayer(input_dim=input_dim, hidden_dim=hidden_dim)

        if number_forward_layers > 1:
            self.feed_forward_rest_vet = [ForwardLayer(input_dim=hidden_dim, hidden_dim=hidden_dim) for _ in
                                          np.arange(number_forward_layers - 1)]

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



# Not in use now.
####################################################################
# Functions to generate data from the argmax of a Gaussian process #
####################################################################
# def argmax_gaussian_process_one_trial_one_net(_, z_mat, network_model_class, network_model_class_kwargs,
#                                               network_net_size):
#     """
#     Simulate a single Gaussian process on data of a single trial.
#
#     :param _: A dummy argument. It allows the function to be called inside the multiprocess Pool.map method.
#     :param z_mat: An n by p dimension numpy array or tensor. n is the sample size and p is the dimension.
#             This is the data we condition on.
#     :param network_model_class: A subclass of tf.keras.Model with output dimension 3. An instance of the class is the
#         neural network to fit on the data.
#     :param network_model_class_kwarg: A dictionaries which contains keyword arguments to create an instance of the
#         network_model_class.
#     :param network_net_size: The number of neural networks we use to create the epsilon net.
#
#     :return:
#         The test statistic which use the argmax of the Gaussian process to compute.
#     """
#     variance_vet = np.zeros(network_net_size)
#     for i in range(network_net_size):
#         network_model = network_model_class(**network_model_class_kwargs)
#         jxy_vet = network_model(z_mat)[:, 2]
#
#         # Notice that the variance is actually the test statistic.
#         variance_vet[i] = np.mean(jxy_vet ** 2)
#
#     gaussian_process_vet = np.random.normal(scale=variance_vet, size=network_net_size)
#     argmax = np.argmax(gaussian_process_vet)
#
#     return variance_vet[argmax]
#
#
# def argmax_gaussian_process_one_trial(pool, z_mat, network_model_class, network_model_class_kwargs, network_net_size,
#                                       number_of_nets):
#     """
#     Return a sample of test statistic by repeating one_trial_one_net {number_of_nets} times.
#
#     :param pool: A multiprocessing.pool.Pool instance.
#     :param z_mat: See one_trial_one_net function.
#     :param network_model_class: See one_trial_one_net function.
#     :param network_model_class_kwargs: See one_trial_one_net function.
#     :param network_net_size: See one_trial_one_net function.
#     :param number_of_nets: An integer. The number of different epsilon net to generated.
#
#     :return:
#         A numpy array of length {number_of_nets}.
#     """
#     test_statistic_sample_vet = pool.map(partial(argmax_gaussian_process_one_trial_one_net, z_mat=z_mat,
#                                                  network_model_class=network_model_class,
#                                                  network_model_class_kwargs=network_model_class_kwargs,
#                                                  network_net_size=network_net_size), np.arange(number_of_nets))
#
#     return np.array(test_statistic_sample_vet)