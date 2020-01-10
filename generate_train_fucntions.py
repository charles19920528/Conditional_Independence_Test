import tensorflow as tf
# Numpy is for the generate_x_y_mat method and trainning method in the IsingSimulation class.
import numpy as np
import hyperparameters as hp

class IsingNetwork(tf.keras.Model):
    def __init__(self, input_dim, hidden_1_out_dim, output_dim):
        super().__init__(input_dim, hidden_1_out_dim, output_dim)

        self.input_dim = input_dim
        self.hidden_1_out_dim = hidden_1_out_dim
        self.output_dim = hidden_1_out_dim

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
        dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1, dtype = tf.float32)
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
        ], dtype = tf.float64)

    elif number_of_columns == 3:
        one_mat = tf.constant([
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype = tf.float64)

    else:
        raise Exception("The shape of the parameter_mat doesn't satisfy the requirement")

    parameter_mat = tf.cast(parameter_mat, tf.float64)
    kernel_mat = tf.matmul(parameter_mat, one_mat, transpose_b=True)
    exp_kernel_mat = tf.exp(kernel_mat)
    prob_mat = tf.transpose(exp_kernel_mat) / tf.reduce_sum(exp_kernel_mat, axis=1)

    return tf.transpose(prob_mat)


def kl_divergence(true_parameter_mat, predicted_parameter_mat, isAverage):
    """
    Compute the average or individual KL divergence between two sets of Ising models. KL(True distribution,
    Fitted distribution) (/ n). See Wikipedia for detail description.
    :param true_parameter_mat: an n by p tensor storing parameters for the true distribution. Each row contains a
    parameter for a one sample. If p = 2, we assume the sample is under the null model. If p = 3, we assume the
    sample is under the full model.
    :param predicted_parameter_mat: an n by p tensor storing parameters for the fitted distribution. We again assume
    that p can either be 2 or 3.
    :param isAverage: a boolean value. If it is true, then the function will return the average kl divergence of the
    sample. Otherwise, it will return the kl divergence between distribiutions for each sample (Z)_.
    :return: kl_divergence_scalr: a scalar
    or kl_divergence_list: a list
    """
    pmf_mat_true = pmf_collection(true_parameter_mat)
    pmf_mat_prediction = pmf_collection(predicted_parameter_mat)

    kl_divergence_mat = pmf_mat_true * tf.math.log(pmf_mat_true / pmf_mat_prediction)
    if isAverage:
        kl_divergence_scalar = tf.reduce_sum(kl_divergence_mat) / true_parameter_mat.shape[0]
        return kl_divergence_scalar
    else:
        kl_divergence_list = tf.reduce_sum(kl_divergence_mat, axis = 1)
        return kl_divergence_list

#################################
# Functions used under the null #
#################################
def pmf_null(x, hx):
    """
    Compute the probability P(X = x) under the null model.
    :param x: Expect either a scalar or an 1-d array of either 1 or negative 1.
    :param hx: The parameter corresponds to x.
    :return: pmf: P(X = x).
    """
    hx = tf.cast(hx, tf.float64)
    numerator = tf.exp(- x * hx)
    denominator = tf.exp(- x * hx) + tf.exp(x * hx)
    pmf = numerator / denominator

    return pmf


def log_ising_null(x_y_mat, parameter_mat):
    """
    Compute - log likelihood of the Ising model under the null and use it as the loss function for model training.
    :param x_y_mat: an n by 2 tensor which stores observed x's any y's.
    :param parameter_mat: a tensor of shape [n, 2]. It should be the output of an object of class IsingNetwork.
    :return: negative_log_likelihood
    """
    dot_product_sum = tf.reduce_sum(x_y_mat * parameter_mat)

    normalizing_constant = 0
    for i in tf.range(parameter_mat.shape[0]):
        # Extract the ith row of the parameter_mat and change it into a (2, 1) tensor.
        j_vet = parameter_mat[i, :][..., None]
        one_vet = tf.constant([1, -1], dtype = tf.float32)[None, ...]
        outer_product = j_vet * one_vet
        log_sum_exp_vet = tf.reduce_logsumexp(outer_product, 1)
        normalizing_constant_i = tf.reduce_sum(log_sum_exp_vet)
        normalizing_constant += normalizing_constant_i

    negative_log_likelihood = dot_product_sum + normalizing_constant
    return negative_log_likelihood


########################################
# Functions used under the alternative #
########################################
def log_ising_alternative(x_y_mat, parameter_mat):
    """
    Compute - log likelihood of the Ising model under the alternative and use it as the loss function for
    model training.
    :param x_y_mat: an n by 2 tensor which stores observed x's any y's.
    :param parameter_mat: a tensor of shape [n, 3]. It should be the output of an object of class IsingNetwork.
    Columns of the matrices are Jx, Jy and Jxy respectively.
    :return: negative_log_likelihood
    """
    parameter_mat = tf.cast(parameter_mat, tf.float32)

    x_times_y = x_y_mat[:, 0] * x_y_mat[:, 1]
    x_times_y = tf.reshape(x_times_y, (tf.shape(x_times_y)[0], 1))
    x_y_xy_mat = tf.concat(values = [x_y_mat, x_times_y], axis = 1)
    x_y_xy_mat = tf.dtypes.cast(x_y_xy_mat, tf.float32)
    dot_product_sum = tf.reduce_sum(x_y_xy_mat * parameter_mat)

    normalizing_constant = tf.constant(0., dtype = tf.float32)
    for i in tf.range(tf.shape(parameter_mat)[0]):
        parameter_vet = parameter_mat[i, :]
        one_mat = tf.constant([
            [-1., -1., -1.],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype = tf.float32)
        exponent_vet = tf.reduce_sum(parameter_vet * one_mat, axis=1)
        log_sum_exp = tf.reduce_logsumexp(exponent_vet)
        normalizing_constant += log_sum_exp

    negative_log_likelihood = dot_product_sum + normalizing_constant
    return negative_log_likelihood

######################################
# Function used to generate the data #
######################################
def generate_x_y_mat(ising_network, z_mat, null_boolean, sample_size):
    """
    The function will generate the matrix of responses (x, y).
    :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
    Usually, it is the output of the generate_z_mat method.
    :return: x_y_mat
    """
    true_parameter_mat = ising_network(z_mat)
    if null_boolean:
        p_equal_1_mat = pmf_null(1, true_parameter_mat)
        x_y_mat = np.random.binomial(n=1, p=p_equal_1_mat, size=(sample_size, 2)).astype(np.float32) * 2 - 1
        return x_y_mat
    else:
        p_mat = pmf_collection(true_parameter_mat)
        # Recall that the column of p_mat corresponds to P(X = 1, Y = 1),
        # P(X = 1, Y = -1), P(X = -1, Y = 1) and P(X = -1, Y = -1)
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

def data_generate_network(dim_z = hp.dim_z, hidden_1_out_dim = hp.hidden_1_out_dim):
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

#########################################
# Class for the simulation and training #
#########################################
class IsingSimulation:
    def __init__(self, z_mat, true_network, null_boolean, hidden_1_out_dim,learning_rate, buffer_size, batch_size,
                 epoch):
        """
        Create a class which can generate data and train a network. It is used to get training oracle information
        such as training epoch.
        :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
        :param true_network: An object of IsingNetwork class. This is the true network we use to generate parameters.
        :param null_boolean: A boolean value indicating if the true model is under the null.
        :param hidden_1_out_dim: A scalar which is the output dimension of the hidden layer.
        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.batch function.
        :param epoch: A scalar indicating the number of times training process pass through the data set.
        """
        self.z_mat = z_mat
        self.sample_size = z_mat.shape[0]
        self.true_network = true_network
        self.null_boolean = null_boolean
        self.hidden_1_out_dim = hidden_1_out_dim
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch = epoch

        self.true_parameter_mat = true_network(z_mat)

    def generate_x_y_mat(self):
        """
        The function will generate the matrix of responses (x, y).
        :param null_boolean: A boolean value indicating if we are simulating under the the independence assumption.
        :return: x_y_mat
        """
        true_parameter_mat = self.true_network(self.z_mat)
        if self.null_boolean:
            p_equal_1_mat = pmf_null(1, true_parameter_mat)
            x_y_mat = np.random.binomial(n=1, p=p_equal_1_mat, size=(self.sample_size, 2)).astype("float32") * 2 - 1
            return x_y_mat
        else:
            p_mat = pmf_collection(true_parameter_mat)
            # Recall that the column of p_mat corresponds to P(X = 1, Y = 1),
            # P(X = 1, Y = -1), P(X = -1, Y = 1) and P(X = -1, Y = -1)
            raw_sample_mat = np.zeros((self.sample_size, 4))
            for i in np.arange(self.sample_size):
                p_vet = p_mat[i, :]
                raw_sample = np.random.multinomial(1, p_vet)
                raw_sample_mat[i, :] = raw_sample
            conversion_mat = np.array([
                [1, 1], [1, -1], [-1, 1], [-1, -1]
            ])
            x_y_mat = raw_sample_mat.dot(conversion_mat)

            return x_y_mat


    def trainning(self, x_y_mat, print_loss_boolean):
        """
        Train a neural network.
        :param x_y_mat: An n x 2 numpy array. Each row is the response of the ith observation.
        First column corresponds to x.
        :param print_loss_boolean: A boolean value dictating if the method will print loss during training.
        :return: result_dict: A dictionary which contains two keys which are "loss_array" and "ising_par".
        result_dict["loss_array"] is a 2 by epoch numpy of which the first row stores the (- 2 * LogLikelihood) and the
        second row stores the kl divergences. result_dict["ising_parameters"] stores a tensor which is the fitted value
        of parameters in the full Ising Model.
        """
        # Prepare training data.
        train_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, x_y_mat))
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)

        # Prepare storage for results.
        loss_kl_array = np.zeros((2, self.epoch))
        result_dict = dict()

        train_network = IsingNetwork(self.z_mat.shape[1], self.hidden_1_out_dim, 3)
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = train_network(z_batch)
                    loss = log_ising_alternative(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, train_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

                predicted_parameter_mat = train_network(self.z_mat)
                batch_kl = kl_divergence(self.true_parameter_mat, predicted_parameter_mat, True)

            if i % 5 == 0 and print_loss_boolean:
                print("Sample size %d, Epoch %d" % (self.sample_size, i))
                print("The loss is %f " % loss)
                print("The KL divergence is %f" % batch_kl)

            loss_kl_array[0, i] = loss.numpy()
            loss_kl_array[1, i] = batch_kl

        result_dict["loss_array"] = loss_kl_array
        result_dict["ising_parameters"] = predicted_parameter_mat

        return result_dict


class IsingTrainingPool:
    def __init__(self, z_mat, epoch, hidden_1_out_dim=hp.hidden_1_out_dim, learning_rate=hp.learning_rate,
                 buffer_size=hp.buffer_size, batch_size=hp.batch_size):
        """
        Create a class which can generate data and train a network.
        :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
        :param hidden_1_out_dim: A scalar which is the output dimension of the hidden layer.
        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.batch function.
        :param epoch: A scalar indicating the number of times training process pass through the data set.
        """
        self.z_mat = z_mat
        self.hidden_1_out_dim = hidden_1_out_dim
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch = epoch

        self.sample_size = z_mat.shape[0]
        self.dim_z = z_mat.shape[1]


    def trainning(self, x_y_mat):
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
        train_ds = tf.data.Dataset.from_tensor_slices((self.z_mat, x_y_mat))
        train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size)

        train_network = IsingNetwork(self.dim_z, self.hidden_1_out_dim, 3)
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = train_network(z_batch)
                    loss = log_ising_alternative(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, train_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

        predicted_parameter_mat = train_network.predict(self.z_mat)

        return predicted_parameter_mat




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

        merged_dataset = tf.data.Dataset.from_generator(dt_iterator, output_types = (tf.float32, tf.float32),
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
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        batch_predicted_parameter_mat = tf.Variable(tf.zeros((self.batch_size, 3), dtype=tf.float32), trainable = True)
        loss = tf.Variable(0, dtype=tf.float32, trainable = True)
        predicted_parameter_mat = tf.Variable(tf.zeros((self.sample_size, 3), dtype=tf.float32))

        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat.assign(train_network(z_batch))
                    loss.assign(log_ising_alternative(x_y_batch, batch_predicted_parameter_mat))
                grads = tape.gradient(loss, train_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))


        """
        for i in range(self.epoch):
            for z_batch, x_y_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_predicted_parameter_mat = train_network(z_batch)
                    loss = log_ising_alternative(x_y_batch, batch_predicted_parameter_mat)
                grads = tape.gradient(loss, train_network.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

        predicted_parameter_mat = tf.Variable( tf.zeros( (self.sample_size, 3), dtype = tf.float32) )
        """

        for z_mat in self.z_dataset.batch(self.sample_size):
            predicted_parameter_mat.assign(train_network(z_mat))


        #predicted_parameter_mat = train_network.predict(self.z_dataset.batch(self.batch_size))

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
