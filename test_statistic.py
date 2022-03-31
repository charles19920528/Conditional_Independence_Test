import numpy as np
import tensorflow as tf
import generate_train_functions as gt


def bias_dot(x_without_bias_mat: np.array, weights_with_bias_vet: np.array):
    """
    Compute X Beta + bias.

    :param x_without_bias_mat: n by p numpy array
    :param weights_with_bias_vet: length p + 1 vet. The last element is the intercept.

    :return:
        A length n numpy array.
    """
    return x_without_bias_mat.dot(weights_with_bias_vet[:-1]) + weights_with_bias_vet[-1]


# x_mat = np.arange(6).reshape(2, 3)
# weights = np.array([1, 1, 0, 10])
# bias_dot(x_mat, weights)


# def theta_derivatives(x_y_mat, final_linear_input_mat, j_mat, gradient_boolean, hessian_boolean):
#     if not (gradient_boolean or hessian_boolean):
#         return None
#     result_list = []
#
#     e_jx_p_jy_vet = np.exp(2 * (j_mat[:, 0] + j_mat[:, 1]))
#     e_jx_p_jxy_vet = np.exp(2 * (j_mat[:, 0] + j_mat[:, 2]))
#     e_jy_p_jxy_vet = np.exp(2 * (j_mat[:, 1] + j_mat[:, 2]))
#     e1_denominator_vet = e_jx_p_jy_vet + e_jx_p_jxy_vet + e_jy_p_jxy_vet + 1
#
#     if gradient_boolean:
#         x_times_y_vet = x_y_mat[:, 0] * x_y_mat[:, 1]
#         multiplier_vet = 2 * (e_jx_p_jy_vet + 1) / e1_denominator_vet - x_times_y_vet - 1
#         multiplier_vet = multiplier_vet.reshape(-1, 1)
#         # n by p
#         gradient_mat = final_linear_input_mat * multiplier_vet
#         gradient_mat = np.hstack([gradient_mat, multiplier_vet])
#         result_list.append(gradient_mat)
#
#         del multiplier_vet
#
#     if hessian_boolean:
#         cosh_sum_vet = np.cosh(2 * j_mat[:, 0]) + np.cosh(2 * j_mat[:, 1])
#         numerator_vet = 8 * cosh_sum_vet * np.exp(2 * j_mat.sum(axis=1))
#         multiplier_vet = - numerator_vet / np.square(e1_denominator_vet)
#         multiplier_vet = multiplier_vet.reshape(-1, 1)
#
#         hessian_mat = np.zeros([final_linear_input_mat.shape[1] + 1, final_linear_input_mat.shape[1] + 1])
#         for i in np.arange(hessian_mat.shape[0]):
#             for j in np.arange(i, hessian_mat.shape[0]):
#                 second_derivative_vet = multiplier_vet
#                 if i != hessian_mat.shape[0] - 1:
#                     second_derivative_vet = second_derivative_vet * final_linear_input_mat[:, i]
#                 if j != hessian_mat.shape[0] - 1:
#                     second_derivative_vet = second_derivative_vet * final_linear_input_mat[:, j]
#                 hessian_mat[i, j] = hessian_mat[j, i] = second_derivative_vet.mean()
#
#         result_list.append(hessian_mat)
#
#     return result_list


def theta_derivatives(x_y_mat, final_linear_input_mat, j_mat, gradient_boolean, hessian_boolean):
    if not (gradient_boolean or hessian_boolean):
        return None
    result_list = []

    e_jx_p_jy_vet = np.exp(2 * (j_mat[:, 0] + j_mat[:, 1]))
    e_jx_p_jxy_vet = np.exp(2 * (j_mat[:, 0] + j_mat[:, 2]))
    e_jy_p_jxy_vet = np.exp(2 * (j_mat[:, 1] + j_mat[:, 2]))
    e1_denominator_vet = e_jx_p_jy_vet + e_jx_p_jxy_vet + e_jy_p_jxy_vet + 1

    if gradient_boolean:
        x_times_y_vet = x_y_mat[:, 0] * x_y_mat[:, 1]
        multiplier_vet = 2 * (e_jx_p_jy_vet + 1) / e1_denominator_vet - x_times_y_vet - 1
        multiplier_vet = multiplier_vet.reshape(-1, 1)
        # n by p
        gradient_mat = final_linear_input_mat * multiplier_vet
        gradient_mat = np.hstack([gradient_mat, multiplier_vet])
        result_list.append(gradient_mat)

        del multiplier_vet

    if hessian_boolean:
        cosh_sum_vet = np.cosh(2 * j_mat[:, 0]) + np.cosh(2 * j_mat[:, 1])
        numerator_vet = 8 * cosh_sum_vet * np.exp(2 * j_mat.sum(axis=1))
        multiplier_vet = - numerator_vet / np.square(e1_denominator_vet)
        multiplier_vet = multiplier_vet.reshape(-1, 1, 1)

        final_linear_input_mat = np.concatenate([final_linear_input_mat, np.ones((final_linear_input_mat.shape[0], 1))],
                                                axis=1)
        # idx_vet = np.arange(final_linear_input_mat.shape[0])
        # final_outer_vet = np.multiply.outer(final_linear_input_mat, final_linear_input_mat)[idx_vet, :, idx_vet, :]
        final_outer_vet = final_linear_input_mat[:, :, np.newaxis] * final_linear_input_mat[:, np.newaxis, :]
        hessian_mat_vet = final_outer_vet * multiplier_vet

        result_list.append(hessian_mat_vet.sum(axis=0))

    return result_list


def wald_test(x_y_mat, z_mat, network_test_args_dict: dict, network_weights_vet, sandwich_boolean):
    network_test = gt.FullyConnectedNetworkTest(**network_test_args_dict)
    network_test.call(z_mat[:2, :])
    network_test.set_weights(network_weights_vet)
    theta_vet = np.concatenate([network_weights_vet[-2][:, 2], [network_weights_vet[-1][2]]]).reshape(-1, 1)

    j_mat = network_test.call(z_mat).numpy()
    final_linear_input_mat = network_test.final_linear_input_mat

    gradient_mat, hessian_mat = theta_derivatives(x_y_mat=x_y_mat, final_linear_input_mat=final_linear_input_mat,
                                                  j_mat=j_mat, gradient_boolean=True, hessian_boolean=True)
    if sandwich_boolean:
        inverse_hessian_mat = np.linalg.pinv(hessian_mat)
        gradient_cov_mat = np.cov(gradient_mat.T, ddof=0)

        meat_mat = inverse_hessian_mat.dot(gradient_cov_mat).dot(inverse_hessian_mat)
        # meat_inverse_mat = np.linalg.solve(meat_mat, np.identity(meat_mat.shape[0]))
        meat_inverse_mat = np.linalg.pinv(meat_mat)
        test_statistic = theta_vet.T.dot(meat_inverse_mat).dot(theta_vet)[0, 0]
    else:
        test_statistic = theta_vet.T.dot(-hessian_mat).dot(theta_vet)[0, 0]

    return test_statistic


def vectorized_cov(data_array):
    """
    https://stackoverflow.com/questions/40394775/vectorizing-numpy-covariance-for-3d-array
    :param data_array: An n_trials by sample_size by par_dim array.
    :return:
        A n_trials by par_dim by par_dim array.
    """
    sample_size = data_array.shape[1]
    m1 = data_array - data_array.sum(1, keepdims=True) / sample_size
    out = np.einsum('ijl,ijk->ikl', m1, m1) / sample_size
    return out


class WaldTest:
    def __init__(self, x_y_mat, j_mat, final_linear_input_mat, network_weights_vet, sandwich_boolean):
        self.x_y_mat = x_y_mat

        self.theta_vet = np.concatenate([network_weights_vet[-2][:, 2], [network_weights_vet[-1][2]]]).reshape(-1, 1)

        self.j_mat = j_mat
        self.final_linear_input_mat = final_linear_input_mat
        self.sandwich_boolean = sandwich_boolean

        self.test_statistic = None
        self._bootstrap_test_statistic_vet = None
        self.hessian_mat = None
        self.inverse_hessian_mat = None

    def _compute_test_statistic(self):
        self.gradient_mat, self.hessian_mat = theta_derivatives(x_y_mat=self.x_y_mat,
                                                                final_linear_input_mat=self.final_linear_input_mat,
                                                                j_mat=self.j_mat, gradient_boolean=True,
                                                                hessian_boolean=True)
        if self.sandwich_boolean:
            self.inverse_hessian_mat = np.linalg.pinv(self.hessian_mat)
            gradient_cov_mat = np.cov(self.gradient_mat.T, ddof=0)

            meat_mat = self.inverse_hessian_mat.dot(gradient_cov_mat).dot(self.inverse_hessian_mat)
            # meat_inverse_mat = np.linalg.solve(meat_mat, np.identity(meat_mat.shape[0]))
            meat_inverse_mat = np.linalg.pinv(meat_mat)
            self.test_statistic = self.theta_vet.T.dot(meat_inverse_mat).dot(self.theta_vet)[0, 0] * self.j_mat.shape[0]
        else:
            self.test_statistic = self.theta_vet.T.dot(-self.hessian_mat).dot(self.theta_vet)[0, 0] * \
                                  self.j_mat.shape[0]

    def get_test_statistic(self):
        if self.test_statistic is None:
            self._compute_test_statistic()
        return self.test_statistic

    def p_value(self, n_trials):
        if self.test_statistic is None:
            self._compute_test_statistic()

        # n_trials by sample_size
        noise_mat = np.random.binomial(1, 0.5, (n_trials, self.x_y_mat.shape[0]))
        # gradient is sample_size by par_dim, result is n_trials by par_dim by sample_size
        perturbed_grad_array = self.gradient_mat.T * noise_mat[:, np.newaxis, :]
        # change to n_trials by sample_size by par_dim
        perturbed_grad_array = np.swapaxes(perturbed_grad_array, 1, 2)
        # n_trials by 1 by par_dim
        sum_perturbed_grad_array = perturbed_grad_array.sum(axis=1, keepdims=True)
        if self.sandwich_boolean:
            # n_trials by par_dim by par_dim
            gradient_cov_array = vectorized_cov(data_array=perturbed_grad_array)
            meat_mat = np.einsum("ji, bil -> bjl", self.inverse_hessian_mat, gradient_cov_array)
            meat_mat = meat_mat.dot(self.inverse_hessian_mat)
            # n_trials by par_dim by par_dim
            meat_inverse_mat = np.linalg.pinv(meat_mat)
            self._bootstrap_test_statistic_vet = np.einsum("lij, ljk -> lik", sum_perturbed_grad_array,
                                                           meat_inverse_mat)
            self._bootstrap_test_statistic_vet = np.einsum("loj, lpj -> l", self._bootstrap_test_statistic_vet,
                                                           sum_perturbed_grad_array)
        else:
            self._bootstrap_test_statistic_vet = sum_perturbed_grad_array.dot(-self.hessian_mat)
            self._bootstrap_test_statistic_vet = \
                (self._bootstrap_test_statistic_vet.squeeze() * sum_perturbed_grad_array.squeeze()).sum(axis=1)
