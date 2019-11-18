import tensorflow as tf
import numpy as np

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
        dummy_z = tf.random.normal(shape=(1, self.input_dim), mean=0, stddev=1)
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
        ], dtype="float32")

    elif number_of_columns == 3:
        one_mat = tf.constant([
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype="float32")

    else:
        raise Exception("The shape of the parameter_mat doesn't satisfy the requirement")

    kernel_mat = tf.matmul(parameter_mat, one_mat, transpose_b=True)
    exp_kernel_mat = tf.exp(kernel_mat)
    prob_mat = tf.transpose(exp_kernel_mat) / tf.reduce_sum(exp_kernel_mat, axis=1)

    return tf.transpose(prob_mat)


def kl_divergence(true_parameter_mat, predicted_parameter_mat):
    """
    Compute the average KL divergence between two sets of Ising models. KL(True distribution, Fitted distribution) / n.
    See Wikipedia for detail description.
    :param true_parameter_mat: an n by p tensor storing parameters for the true distribution. Each row contains a
    parameter for a one sample. If p = 2, we assume the sample is under the null model. If p = 3, we assume the
    sample is under the full model.
    :param predicted_parameter_mat: an n by p tensor storing parameters for the fitted distribution. We again assume
    that p can either be 2 or 3.
    :return: kl_divergence: a scalar.
    """
    pmf_mat_true = pmf_collection(true_parameter_mat)
    pmf_mat_prediction = pmf_collection(predicted_parameter_mat)

    kl_divergence_mat = pmf_mat_true * tf.math.log(pmf_mat_true / pmf_mat_prediction)
    kl_divergence = tf.reduce_sum(kl_divergence_mat) / true_parameter_mat.shape[0]

    return kl_divergence

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
        one_vet = tf.constant([1, -1], dtype="float32")[None, ...]
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

    x_times_y = x_y_mat[:, 0] * x_y_mat[:, 1]
    # x_times_y = tf.reshape(x_times_y, (x_times_y.shape[0], 1)) won't work. Weird... Check the reason later.
    x_times_y_reshape = tf.reshape(x_times_y, (x_times_y.shape[0], 1))
    x_y_xy_mat = tf.concat(values = [x_y_mat, x_times_y_reshape], axis = 1)
    dot_product_sum = tf.reduce_sum(x_y_xy_mat * parameter_mat)

    normalizing_constant = 0
    for i in tf.range(parameter_mat.shape[0]):
        parameter_vet = parameter_mat[i, :]
        one_mat = tf.constant([
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype = "float32")
        exponent_vet = tf.reduce_sum(parameter_vet * one_mat, axis=1)
        log_sum_exp = tf.reduce_logsumexp(exponent_vet)
        normalizing_constant += log_sum_exp

    negative_log_likelihood = dot_product_sum + normalizing_constant
    return negative_log_likelihood


#########################################
# Class for the simulation and training #
#########################################
class IsingSimulation:
    def __init__(self, z_mat, true_network, null_boolean, hidden_1_out_dim,learning_rate, buffer_size, batch_size,
                 epoch):
        """
        Create a class which can generate data and train a network.
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
                batch_kl = kl_divergence(self.true_parameter_mat, predicted_parameter_mat)

            if i % 5 == 0 and print_loss_boolean:
                print("Sample size %d, Epoch %d" % (self.sample_size, i))
                print("The loss is %f " % loss)
                print("The KL divergence is %f" % batch_kl)

            loss_kl_array[0, i] = loss.numpy()
            loss_kl_array[1, i] = batch_kl

        result_dict["loss_array"] = loss_kl_array
        result_dict["ising_parameters"] = predicted_parameter_mat

        return result_dict


####################
# Results analysis #
####################
