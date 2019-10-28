import numpy as np
import tensorflow as tf
import generate_train as gt


####################
# Hyper parameters #
####################
hidden_1_out_dim = 3
dim_z = 3
sample_size_vet = [30, 100, 500, 1000]

#######################
# Create null network #
#######################
np.random.seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = np.random.normal(1, 1, size = (dim_z, hidden_1_out_dim))
linear_1_bias_array = np.zeros(shape = (hidden_1_out_dim, ))

linear_2_weight_array = np.random.normal(1, 1, size = (hidden_1_out_dim, 2))
linear_2_bias_array = np.zeros(shape = (2, ))

null_network_generate.set_weights([ linear_1_weight_array, linear_1_bias_array,
                                    linear_2_weight_array, linear_2_bias_array ])

##############
# Generate z #
##############
"""
np.random.seed(1)
for sample_size in sample_size_vet:
    z = np.random.normal(0, 10, (sample_size, dim_z))
    np.savetxt("./data/z_%d.txt" % sample_size, z)

    parameter_mat = null_network_generate(z)
    p_equal_1_mat = gt.pmf_null(1, parameter_mat)
    np.savetxt("./data/p_equal_1_mat_%d.txt" % sample_size, p_equal_1_mat)

    x_y_mat = np.random.binomial(n = 1, p = p_equal_1_mat, size = (sample_size, 2)).astype("float32") * 2 -1
    np.savetxt("./data/x_y_mat_%d.txt" % sample_size, x_y_mat)
"""


####################
# Generate x and y #
####################
"""
This block is left for debuging purposes.
z = np.loadtxt("./data/z_30.txt", dtype = "float32")
parameter_mat = null_network_generate(z)

# We let the first column of parameter_mat correspond to Jx.
p_equal_1_mat = gt.pmf_null(1, parameter_mat)
#p_equal_1_mat[:, 1] = np.ones(30) This line is to check the broadcasting in np.binomial is correct.

# First column corresponds to x.
x_y_mat = np.random.binomial(n = 1, p = p_equal_1_mat, size = (sample_size_vet[0], 2)).astype("float32") * 2 -1
"""



####################
# Network Training #
####################
z = np.loadtxt("./data/z_30.txt", dtype = "float32")
x_y_mat = np.loadtxt("./data/x_y_mat_30.txt", dtype = "float32")

null_network_train = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_train.dummy_run()
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.00001, momentum = True)
# optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
for i in range(200):
    with tf.GradientTape() as tape:
        parameter_mat_pred = null_network_train(z)
        loss = gt.log_ising_null(x_y_mat, parameter_mat_pred)
    grads = tape.gradient(loss, null_network_train.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, null_network_train.variables))

    if i % 5 == 0:
        print("Iteration %d, the loss is %f " % (i, loss))


predicted_parameter = null_network_train(z)
predicted_p_equal_1_mat = gt.pmf_null(1, predicted_parameter)

p_equal_1_mat = np.loadtxt("./data/p_equal_1_mat_%d.txt" % 30)
predicted_p_equal_1_mat - p_equal_1_mat

