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
    Compute the conditoinal pmf for the mixture data. Under the null,

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
        p_mat = np.tile([0, 0.5, 0.5, 0], (nrow, 1))
        helper_pmf_vet = np.array([0.5, 0, 0, 0.5])
        p_mat[~less_than_cut_off_boolean] = np.tile(helper_pmf_vet, (sum(~less_than_cut_off_boolean), 1))

        return p_mat