import numpy as np
import tensorflow as tf
import generate_train_functions as gt


# def bias_dot(x_without_bias_mat: np.array, weights_with_bias_vet: np.array):
#     """
#     Compute X Beta + bias.
#
#     :param x_without_bias_mat: n by p numpy array
#     :param weights_with_bias_vet: length p + 1 vet. The last element is the intercept.
#
#     :return:
#         A length n numpy array.
#     """
#     return x_without_bias_mat.dot(weights_with_bias_vet[:-1]) + weights_with_bias_vet[-1]


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
        result_list.append(-gradient_mat)

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

        result_list.append(-hessian_mat_vet.mean(axis=0))

    return result_list


def vectorized_cov(data_array):
    """
    https://stackoverflow.com/questions/40394775/vectorizing-numpy-covariance-for-3d-array
    :param data_array: An n_trials by sample_size by par_dim array.
    :return:
        A n_trials by par_dim by par_dim array.
    """
    sample_size = data_array.shape[1]
    # m1 = data_array - data_array.sum(1, keepdims=True) / sample_size
    # out = np.einsum('ijl,ijk->ikl', m1, m1) / sample_size
    out = np.einsum('ijl,ijk->ikl', data_array, data_array) / sample_size
    return out


class ScoreTest:
    def __init__(self, x_y_mat, j_mat, final_linear_input_mat, network_weights_vet, sandwich_boolean):
        self.x_y_mat = x_y_mat

        self.j_mat = j_mat
        if self.j_mat.shape[1] == 2:
            self.test_type = "score"
            self.j_mat = np.hstack([self.j_mat, np.zeros([self.j_mat.shape[0], 1])])
        else:
            self.test_type = "wald"
            self.theta_vet = np.concatenate([network_weights_vet[-2][:, 2], [network_weights_vet[-1][2]]]).reshape(-1,
                                                                                                                   1)
        self.final_linear_input_mat = final_linear_input_mat
        self.sandwich_boolean = sandwich_boolean

        self.test_statistic = None
        self.hessian_mat = None
        self.inverse_hessian_mat = None
        self.boostrap_ts_vet = None

    def _compute_test_statistic(self):
        self.gradient_mat, self.hessian_mat = theta_derivatives(x_y_mat=self.x_y_mat,
                                                                final_linear_input_mat=self.final_linear_input_mat,
                                                                j_mat=self.j_mat, gradient_boolean=True,
                                                                hessian_boolean=True)
        if self.test_type == "wald":
            outer_bread_vet = self.theta_vet
        else:
            self.inverse_hessian_mat = np.linalg.pinv(self.hessian_mat * self.j_mat.shape[0]) * self.j_mat.shape[0]
            outer_bread_vet = self.inverse_hessian_mat.dot(self.gradient_mat.mean(axis=0)).reshape(-1, 1)
        if self.sandwich_boolean:
            if self.inverse_hessian_mat is None:
                self.inverse_hessian_mat = np.linalg.pinv(self.hessian_mat * self.j_mat.shape[0]) * self.j_mat.shape[0]
            # gradient_cov_mat = np.cov(self.gradient_mat.T, ddof=0)
            gradient_cov_mat = self.gradient_mat.T.dot(self.gradient_mat) / self.j_mat.shape[0]

            var_mat = self.inverse_hessian_mat.dot(gradient_cov_mat).dot(self.inverse_hessian_mat) / self.j_mat.shape[0]
            # meat_inverse_mat = np.linalg.solve(meat_mat, np.identity(meat_mat.shape[0]))
            var_inverse_mat = np.linalg.pinv(var_mat)
            self.test_statistic = outer_bread_vet.T.dot(var_inverse_mat).dot(outer_bread_vet)[0, 0]
        else:
            self.test_statistic = outer_bread_vet.T.dot(self.hessian_mat).dot(outer_bread_vet)[0, 0] * \
                                  self.j_mat.shape[0]

    def get_test_statistic(self):
        if self.test_statistic is None:
            self._compute_test_statistic()
        if self.test_statistic < 0:
            print("Negative test statistic.")
        return self.test_statistic

    def p_value(self, n_batches, batch_size):
        if self.test_statistic is None:
            self._compute_test_statistic()
        if self.inverse_hessian_mat is None:
            self.inverse_hessian_mat = np.linalg.pinv(self.hessian_mat * self.j_mat.shape[0]) * self.j_mat.shape[0]

        bootstrap_test_statistic_vet_list = []
        for _ in range(n_batches):
            # n_trials by sample_size
            # noise_mat = 2 * np.random.binomial(1, 0.5, (n_trials, self.x_y_mat.shape[0])) - 1
            noise_mat = np.random.gamma(shape=4, scale=0.5, size=(batch_size, self.x_y_mat.shape[0])) - 2
            # gradient is sample_size by par_dim, result is n_trials by par_dim by sample_size
            perturbed_grad_array = self.gradient_mat.T * noise_mat[:, np.newaxis, :]
            # change to n_trials by sample_size by par_dim
            perturbed_grad_array = np.swapaxes(perturbed_grad_array, 1, 2)
            # n_trials by 1 by par_dim
            bread_array = perturbed_grad_array.mean(axis=1, keepdims=True)
            bread_array = np.einsum("ij, lnj -> lni", self.inverse_hessian_mat, bread_array)
            if self.sandwich_boolean:
                # n_trials by par_dim by par_dim
                gradient_cov_array = vectorized_cov(data_array=perturbed_grad_array)
                meat_mat = np.einsum("ji, bil -> bjl", self.inverse_hessian_mat, gradient_cov_array)
                meat_mat = meat_mat.dot(self.inverse_hessian_mat) / self.j_mat.shape[0]
                # n_trials by par_dim by par_dim
                meat_inverse_mat = np.linalg.pinv(meat_mat)
                bootstrap_test_statistic_vet = np.einsum("lij, ljk -> lik", bread_array,
                                                         meat_inverse_mat)
                bootstrap_test_statistic_vet = np.einsum("loj, lpj -> l", bootstrap_test_statistic_vet,
                                                         bread_array)
            else:
                bootstrap_test_statistic_vet = bread_array.dot(self.hessian_mat)
                bootstrap_test_statistic_vet = \
                    (bootstrap_test_statistic_vet.squeeze() * bread_array.squeeze()).sum(axis=1) * self.j_mat.shape[0]
            bootstrap_test_statistic_vet_list.append(bootstrap_test_statistic_vet)
            self.boostrap_ts_vet = np.hstack(bootstrap_test_statistic_vet_list)
        return sum(self.boostrap_ts_vet > self.test_statistic) / (n_batches * batch_size)
    #
    # def p_value_beta(self, n_trials):
    #     if self.test_statistic is None:
    #         self._compute_test_statistic()
    #     if self.inverse_hessian_mat is None:
    #         self.inverse_hessian_mat = np.linalg.pinv(self.hessian_mat)
    #
    #     ts_list = []
    #     for _ in np.arange(n_trials):
    #         noise_vet = np.random.gamma(shape=4, scale=0.5, size=(self.x_y_mat.shape[0], 1)) - 2
    #         # noise_vet = np.random.normal(size=(self.x_y_mat.shape[0], 1))
    #         perturbed_grad_array = self.gradient_mat * noise_vet
    #         bread_array = self.inverse_hessian_mat.dot(perturbed_grad_array.mean(axis=0).T)
    #         if self.sandwich_boolean:
    #             gradient_cov_mat = np.cov(perturbed_grad_array.T, ddof=0)
    #             var_mat = self.inverse_hessian_mat.dot(gradient_cov_mat).dot(self.inverse_hessian_mat) / \
    #                       self.j_mat.shape[0]
    #             var_inverse_mat = np.linalg.pinv(var_mat)
    #             bootstrap_test_statistic = bread_array.dot(var_inverse_mat).dot(bread_array)
    #         else:
    #             bootstrap_test_statistic = bread_array.T.dot(-self.hessian_mat).dot(bread_array) * self.j_mat.shape[0]
    #         ts_list.append(bootstrap_test_statistic)
    #
    #
    #     return sum(np.array(ts_list) > self.test_statistic) / n_trials
