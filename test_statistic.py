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
        multiplier_vet = multiplier_vet.reshape(-1, 1)

        hessian_mat = np.zeros([final_linear_input_mat.shape[1] + 1, final_linear_input_mat.shape[1] + 1])
        for i in np.arange(hessian_mat.shape[0]):
            for j in np.arange(i, hessian_mat.shape[0]):
                second_derivative_vet = multiplier_vet
                if i != hessian_mat.shape[0] - 1:
                    second_derivative_vet = second_derivative_vet * final_linear_input_mat[:, i]
                if j != hessian_mat.shape[0] - 1:
                    second_derivative_vet = second_derivative_vet * final_linear_input_mat[:, j]
                hessian_mat[i, j] = hessian_mat[j, i] = second_derivative_vet.mean()

        result_list.append(hessian_mat)

    return result_list


class WaldTest:
    def __init__(self, x_y_mat, z_mat, network_test: gt.FullyConnectedNetworkTest, network_weights_vet):
        self.x_y_mat = x_y_mat
        self.z_mat = z_mat

        self.network_test = network_test
        self.network_test.call(z_mat[0:2, :])
        self.network_test.set_weights(network_weights_vet)

        self.theta_vet = np.concatenate([network_weights_vet[-2][:, 2], [network_weights_vet[-1][2]]]).reshape(-1, 1)

        self.j_mat = self.network_test.call(z_mat).numpy()
        self.final_linear_input_mat = self.network_test.final_linear_input_mat

        self.test_statistic = None

    def _compute_test_statistic(self):
        self.gradient_mat, hessian_mat = theta_derivatives(x_y_mat=self.x_y_mat,
                                                           final_linear_input_mat=self.final_linear_input_mat,
                                                           j_mat=self.j_mat, gradient_boolean=True,
                                                           hessian_boolean=True)
        self.inverse_hessian_mat = np.linalg.solve(hessian_mat, np.identity(hessian_mat.shape[0]))
        gradient_cov_mat = np.cov(self.gradient_mat.T, ddof=0)
        meat_mat = self.inverse_hessian_mat.dot(gradient_cov_mat).dot(self.inverse_hessian_mat)
        meat_inverse_mat = np.linalg.solve(meat_mat, np.identity(meat_mat.shape[0]))
        self.test_statistic = self.theta_vet.T.dot(meat_inverse_mat).dot(self.theta_vet)

    def get_test_statistic(self):
        if self.test_statistic is None:
            self._compute_test_statistic()
        else:
            return self.test_statistic

# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) > 0)