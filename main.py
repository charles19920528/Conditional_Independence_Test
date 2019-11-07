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
"""


#############################
# Fit the alternative model #
#############################
z = np.loadtxt("./data/z_%d.txt" % 30, dtype="float32")
p_equal_1_mat = np.loadtxt("./data/p_equal_1_mat_%d.txt" % 30, dtype="float32")


x_y_mat = np.random.binomial(n = 1, p = p_equal_1_mat, size = (sample_size_vet[0], 2)).astype("float32") * 2 -1
alt_network = gt.IsingNetwork(dim_z, hidden_1_out_dim, 3)
alt_network.dummy_run()

learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
training_iteration = 3000
train_loss = np.zeros(training_iteration)
alt_network(z)

for i in range(training_iteration):
    with tf.GradientTape() as tape:
        parameter_mat_pred = alt_network(z)
        loss = gt.log_ising_alternative(x_y_mat, parameter_mat_pred)
    grads = tape.gradient(loss, alt_network.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, alt_network.variables))

    train_loss[i] = loss.numpy()

    if i % 10 == 0:
        print("Iteration %d, the loss is %f " % (i, loss))



####################
# Network Training #
####################
for training_sample_size in sample_size_vet:
    z = np.loadtxt("./data/z_%d.txt" % training_sample_size, dtype="float32")
    x_y_mat = np.loadtxt("./data/x_y_mat_%d.txt" % training_sample_size, dtype="float32")
    p_equal_1_mat = np.loadtxt("./data/p_equal_1_mat_%d.txt" % training_sample_size, dtype="float32")

    null_network_train = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
    null_network_train.dummy_run()
    # optimizer = tf.keras.optimizers.SGD(learning_rate = 0.00001, momentum = True)
    learning_rate = 0.005
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    iteration = 1600
    train_loss = np.zeros(iteration)
    kl_divergence = np.zeros(iteration)
    for i in range(iteration):
        with tf.GradientTape() as tape:
            parameter_mat_pred = null_network_train(z)
            loss = gt.log_ising_null(x_y_mat, parameter_mat_pred)
        grads = tape.gradient(loss, null_network_train.variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, null_network_train.variables))

        parameter_mat_pred = null_network_train(z)
        p_equal_1_hat_mat = gt.pmf_null(1, parameter_mat_pred)
        kl_divergence[i] = np.sum(p_equal_1_mat * np.log(p_equal_1_mat / p_equal_1_hat_mat))
        train_loss[i] = loss.numpy()

        if i % 10 == 0:
            print("Iteration %d, the loss is %f " % (i, loss))
            print("The KL divergence is %f" % kl_divergence[i])


    plt.figure(training_sample_size)
    plt.plot(train_loss, label = "likelihood")
    plt.plot(kl_divergence, label = "kl")
    plt.legend()
    plt.savefig("./figure/Loss function_%d.png" % training_sample_size)

    np.savetxt("./results/train_loss_%d.txt" % training_sample_size, train_loss)
    np.savetxt("./results/kl_%d.txt" % training_sample_size, kl_divergence)


# Saved for graduate student descent

training_sample_size = 30
z = np.loadtxt("./data/z_%d.txt" % training_sample_size, dtype = "float32")
x_y_mat = np.loadtxt("./data/x_y_mat_%d.txt" % training_sample_size , dtype = "float32")
p_equal_1_mat = np.loadtxt("./data/p_equal_1_mat_%d.txt" % training_sample_size, dtype = "float32")

null_network_train = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_train.dummy_run()
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.00001, momentum = True)

iteration = 800

# Let learning rate decay.
"""
step = tf.Variable(0, trainable = False)
decay_times = int(iteration / 200)
boundaries = np.linspace(0, iteration, num = decay_times + 1).tolist()
values = np.geomspace(10 ** (-2), 10 ** (- decay_times - 3),
                      num = decay_times + 2).tolist()
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
learning_rate = learning_rate_fn(step)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
"""
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)



train_loss = np.zeros(iteration)
kl_divergence = np.zeros(iteration)
for i in range(iteration):
    with tf.GradientTape() as tape:
        parameter_mat_pred = null_network_train(z)
        loss = gt.log_ising_null(x_y_mat, parameter_mat_pred)
    grads = tape.gradient(loss, null_network_train.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, null_network_train.variables))
#    step = step + 1

    parameter_mat_pred = null_network_train(z)
    p_equal_1_hat_mat = gt.pmf_null(1, parameter_mat_pred)
    kl_divergence[i] = np.sum(p_equal_1_mat * np.log(p_equal_1_mat / p_equal_1_hat_mat))
    train_loss[i] = loss.numpy()

    if i % 10 == 0:
        print("Iteration %d, the loss is %f " % (i, loss))
        print("The KL divergence is %f" % kl_divergence[i])






plt.plot(train_loss, label = "likelihood")
plt.plot(kl_divergence, label = "kl")
plt.legend()
plt.savefig("./figure/Loss function_%d.png" % training_sample_size)



