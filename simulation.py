import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import matplotlib.pyplot as plt

####################
# Hyper parameters #
####################
hidden_1_out_dim = 3
dim_z = 3
sample_size_vet = [30, 100, 500, 1000]


##############
# Generate z #
##############
np.random.seed(1)
for sample_size in sample_size_vet:
    z = np.random.normal(0, 10, (sample_size, dim_z))
    np.savetxt("./data/z_%d.txt" % sample_size, z)


###########################
# Simulate under the null #
###########################
# Create null network
np.random.seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = np.random.normal(1, 1, size = (dim_z, hidden_1_out_dim))
linear_1_bias_array = np.zeros(shape = (hidden_1_out_dim, ))

linear_2_weight_array = np.random.normal(1, 1, size = (hidden_1_out_dim, 2))
linear_2_bias_array = np.zeros(shape = (2, ))

null_network_generate.set_weights([ linear_1_weight_array, linear_1_bias_array,
                                    linear_2_weight_array, linear_2_bias_array ])


# Produce p_equal_1_mat.
z = np.loadtxt("./data/z_30.txt", dtype = "float32")
parameter_mat = null_network_generate(z)
p_equal_1_mat = gt.pmf_null(1, parameter_mat)


########## loop over different simulated sample start
# Generate samples
x_y_mat = np.random.binomial(n = 1, p = p_equal_1_mat, size = (30, 2)).astype("float32") * 2 -1

train_network = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
train_network.dummy_run()

learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
epoch = 3000

kl_divergence = training_loss = np.zeros(epoch)

for i in range(epoch):
    with tf.GradientTape() as tape:
        parameter_mat_pred = train_network(z)
        loss = gt.log_ising_null(x_y_mat, parameter_mat_pred)
    grads = tape.gradient(loss, train_network.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))



    training_loss[i] = loss.numpy()

    if i % 10 == 0:
        print("Iteration %d, the loss is %f " % (i, loss))