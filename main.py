import numpy as np
import tensorflow as tf
import generate_train as gt
from z_generation import dim_z, sample_size_vet

####################
# Hyper parameters #
####################
hidden_1_out_dim = 3


#######################
# Create null network #
#######################
np.random.seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = np.random.normal(1, 1, size = (dim_z, hidden_1_out_dim))
# linear_1_bias_array = np.random.normal(0, 2, size = (hidden_1_out_dim, ))
linear_1_bias_array = np.zeros(shape = (hidden_1_out_dim, ))

linear_2_weight_array = np.random.normal(1, 1, size = (hidden_1_out_dim, 2))
# linear_2_bias_array = np.random.normal(0, 1, size = (2, ))
linear_2_bias_array = np.zeros(shape = (2, ))

null_network_generate.set_weights([ linear_1_weight_array, linear_1_bias_array,
                                    linear_2_weight_array, linear_2_bias_array ])


####################
# Generate x and y #
####################
# z = np.loadtxt("./data/z_%d.txt")
z = np.loadtxt("./data/z_30.txt", dtype = "float32")
parameter_mat = null_network_generate(z)

# We let the first column of parameter_mat correspond to Jx.
p_equal_1_mat = gt.pmf_null(1, parameter_mat)
#p_equal_1_mat[:, 1] = np.ones(30)

# First column corresponds to x.
x_y_mat = np.random.binomial(n = 1, p = p_equal_1_mat, size = (sample_size_vet[0], 2)).astype("float32")

#test = gt.log_ising_null(x_y_mat, parameter_mat)


####################
# Network Training #
####################
null_network_train = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_train.dummy_run()
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum = True)
# optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
for i in range(500):
    with tf.GradientTape() as tape:
        parameter_mat_pred = null_network_train(z)
        loss = gt.log_ising_null(x_y_mat, parameter_mat_pred)
    grads = tape.gradient(loss, null_network_train.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, null_network_train.variables))

    if i % 5 == 0:
        print("Iteration %d, the loss is %f " % (i, loss))

predicted_parameter = null_network_train(z)
predicted_p_equal_1_mat = gt.pmf_null(1, predicted_parameter)

predicted_p_equal_1_mat - p_equal_1_mat

null_network_train.variables[0]
null_network_generate.variables[0]



"""
null_network.linear_1.set_weights( [ linear_1_weight_array, [1,2,3]] )
null_model.linear_1.set_weights([ [0,1,2], [2,3,4] ])

null_network.linear_1.get_weights()
null_network.linear_2.get_weights()



class IcingLikelihood(tf.keras.losses.Loss):
    def call(self, y_true):



x = tf.Tensor([1.0,2,3])


 (200 / 239 - 240 /239 * 5**2/ 6**2)

x = np.ones(240)
x[200:239] = 0
np.std(x)

"""

