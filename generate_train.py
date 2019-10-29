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

def log_ising_null(x_y_mat, parameter_mat):
    """
    Compute - log likelihood of the Ising model under the null and use it as the loss function for model training.
    :param x_y_mat: an n by 2 tensor which stores observed x's any y's.
    :param parameter_mat: A tensor of shape [n, 2]. It should be the output of an object of class NullNetwork.
    :return: - LogLikelihood
    """
#    dot_product_sum = tf.reduce_sum( tf.linalg.diag_part( x_y_mat.dot( tf.transpose(parameter_mat) ) ) )
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

    return dot_product_sum + normalizing_constant

def log_ising_full(x_y_mat, parameter_mat):
    pass












"""
class LogIsingNull(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        
       # Compute - log likelihood of the Ising model under the null and use it as the loss function for model training.
       # :param y_true: Expect a n by 2 array which stores observed x's any y's.
       # :param y_pred: A tensor of shape [n, 2]. It should be the output of an object of class NullNetwork.
       # :return: - LogLikelihood
       
        dot_product_sum = tf.reduce_sum( tf.linalg.diag_part( y_true.dot( tf.transpose(y_pred) ) ) )

        normalizing_constant = 0
        for i in np.arange(y_pred.shape[0]):
            j_vet = y_pred[i, :][..., None]
            one_vet = tf.constant([1, -1], dtype = "float32")[None, ...]
            outer_product = j_vet * one_vet
            log_sum_exp_vet = tf.reduce_logsumexp(outer_product, 1)
            normalizing_constant_i = tf.reduce_sum(log_sum_exp_vet)
            normalizing_constant += normalizing_constant_i

        return dot_product_sum + normalizing_constant
"""

