import tensorflow as tf
import numpy as np

class IsingNetwork(tf.keras.Model):
    def __init__(self, input_dim, hidden_1_out_dim, output_dim):
        super().__init__(input_dim, hidden_1_out_dim, output_dim)

        self.input_dim = input_dim
        self.hidden_1_out_dim = hidden_1_out_dim
        self.output_dim = hidden_1_out_dim

        self.linear_1 = tf.keras.layers.Dense(
            units = hidden_1_out_dim,
            input_shape = (input_dim, )
        )
        #self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.03)
        self.linear_2 = tf.keras.layers.Dense(
            units = output_dim,
            input_shape = (hidden_1_out_dim, )
        )


    def call(self, input):
        output = self.linear_1(input)
        #output = self.leakyRelu(output)
        output = tf.keras.activations.tanh(output)
        output = self.linear_2(output)
        return output

    def dummy_run(self):
        """
        This method is to let python initialize the network and weights not just the computation graph.
        :return: None.
        """

        dummy_z = np.random.normal(0, 1, (1, self.input_dim))
        self(dummy_z)


def pmf_null(x, hx):
    """
    Compute the probability P(X = x) under the null model.
    :param x: Expect either a scalar or an 1-d array of either 1 or negative 1.
    :param hx: The parameter corresponds to x.
    :return: pmf: P(X = x).
    """
    numerator = np.exp(- x * hx)
    denominator = np.exp(- x * hx) + np.exp(x * hx)
    pmf = numerator / denominator

    return pmf


def pmf_collection(parameter_mat, null_True):
    """
    Compute P(X = 1, Y = 1), P(X = 1, Y = -1), P(X = -1, Y = 1) and P(X = -1, Y = -1) given by parameters stored
    in the row of parameter mat.
    :param parameter_mat: a tensor with shape [n, 2] or [n, 3]. Each row corresponds to a set of parameters for the
    Ising model.
    :param null_True: a boolean value. If it is true, we compute the pmf using the null model.
    :return: pmf_mat: a tensor of [n, 4]. Each row is a
    """


    dot_product_mat = x_y_mat * parameter_mat

    sample_size = parameter_mat.shape[0]
    normalizing_constant_vet = np.zeros(sample_size)
    for i in np.arange(sample_size):
        parameter_vet = parameter_mat[i, :]
        one_mat = np.array([
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ])
        exponent_vet = tf.reduce_sum(parameter_vet * one_mat, axis=1)
        sum_exp = tf.reduce_sum(tf.exp(exponent_vet))
        normalizing_constant_vet[i] = sum_exp

    joint_probability = tf.exp(dot_product_mat) / normalizing_constant_vet
    return joint_probability






def log_ising_null(x_y_mat, parameter_mat):
    """
    Compute - log likelihood of the Ising model under the null and use it as the loss function for model training.
    :param x_y_mat: an n by 2 tensor which stores observed x's any y's.
    :param parameter_mat: a tensor of shape [n, 2]. It should be the output of an object of class IsingNetwork.
    :return: negative_log_likelihood
    """
    dot_product_sum = tf.reduce_sum(x_y_mat * parameter_mat)

    normalizing_constant = 0
    for i in np.arange(parameter_mat.shape[0]):
        # Extract the ith row of the parameter_mat and change it into a (2, 1) tensor.
        j_vet = parameter_mat[i, :][..., None]
        one_vet = tf.constant([1, -1], dtype="float32")[None, ...]
        outer_product = j_vet * one_vet
        log_sum_exp_vet = tf.reduce_logsumexp(outer_product, 1)
        normalizing_constant_i = tf.reduce_sum(log_sum_exp_vet)
        normalizing_constant += normalizing_constant_i

    negative_log_likelihood = dot_product_sum + normalizing_constant
    return negative_log_likelihood

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
    x_times_y = x_times_y.reshape(-1, 1)
    x_y_xy_mat = np.hstack((x_y_mat, x_times_y))
    dot_product_sum = tf.reduce_sum(x_y_xy_mat * parameter_mat)

    normalizing_constant = 0
    for i in np.arange(parameter_mat.shape[0]):
        parameter_vet = parameter_mat[i, :]
        one_mat = np.array([
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ])
        exponent_vet = tf.reduce_sum(parameter_vet * one_mat, axis = 1)
        log_sum_exp = tf.reduce_logsumexp(exponent_vet)
        normalizing_constant += log_sum_exp

    negative_log_likelihood = dot_product_sum + normalizing_constant
    return negative_log_likelihood


def kl_divergence_null_vs_alt(parameter_mat_null, parameter_mat_alt):
    sample_size = parameter_mat_null.shape[0]

    one_mat = np.array([
        [-1, -1, -1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
    ])

    for i in np.arange(sample_size):











