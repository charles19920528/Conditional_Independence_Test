import numpy as np
import tensorflow as tf
import generate_train as gt
import matplotlib.pyplot as plt

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
# This block is left for debuging purposes.
z = np.loadtxt("./data/z_30.txt", dtype = "float32")
parameter_mat = null_network_generate(z)

# We let the first column correspond to x.
p_equal_1_mat = gt.pmf_null(1, parameter_mat)
# Next two lines are to check the broadcasting in np.binomial is correct.
#p_equal_1_mat[:, 1] = np.ones(30)
#p_equal_1_mat[:10, 0] = 0

# First column corresponds to x.
x_y_mat = np.random.binomial(n = 1, p = p_equal_1_mat, size = (sample_size_vet[0], 2)).astype("float32") * 2 -1
gt.log_ising_null(x_y_mat, p_equal_1_mat)
"""

####################
# Network Training #
####################
z = np.loadtxt("./data/z_100.txt", dtype = "float32")
x_y_mat = np.loadtxt("./data/x_y_mat_100.txt", dtype = "float32")

null_network_train = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_train.dummy_run()
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.00001, momentum = True)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

iteration = 1000
train_loss = np.zeros(iteration)
for i in range(iteration):
    with tf.GradientTape() as tape:
        parameter_mat_pred = null_network_train(z)
        loss = gt.log_ising_null(x_y_mat, parameter_mat_pred)
    grads = tape.gradient(loss, null_network_train.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, null_network_train.variables))

    train_loss[i] = loss.numpy()

    if i % 10 == 0:
        print("Iteration %d, the loss is %f " % (i, loss))


predicted_parameter = null_network_train(z)
predicted_p_equal_1_mat = gt.pmf_null(1, predicted_parameter)

p_equal_1_mat = np.loadtxt("./data/p_equal_1_mat_%d.txt" % 100)
abs_difference = np.abs(predicted_p_equal_1_mat - p_equal_1_mat)
plt.hist(abs_difference, normed = True, bins = 10)


np.mean(np.abs(predicted_p_equal_1_mat - p_equal_1_mat))
np.std(np.abs(predicted_p_equal_1_mat - p_equal_1_mat))