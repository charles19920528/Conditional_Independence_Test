import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import matplotlib.pyplot as plt

hidden_1_out_dim = 3
dim_z = 3

np.random.seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = np.random.normal(1, 1, size=(dim_z, hidden_1_out_dim))
linear_1_bias_array = np.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = np.random.normal(1, 1, size=(hidden_1_out_dim, 2))
linear_2_bias_array = np.zeros(shape=(2,))

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])



sample_size = 30

z = np.loadtxt("./data/z_%d.txt" % sample_size, dtype="float32")
true_parameter_mat = null_network_generate(z)
p_equal_1_mat = gt.pmf_null(1, true_parameter_mat)

# Generate samples
x_y_mat = np.random.binomial(n=1, p=p_equal_1_mat, size=(sample_size, 2)).astype("float32") * 2 - 1


train_dt = tf.data.Dataset.from_tensor_slices((z, x_y_mat))
train_dt = train_dt.shuffle(100).batch(10)
for z_i, x_y in train_dt:
    print(x_y)




train_network = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
train_network.dummy_run()

learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
epoch = 700

kl_divergence = np.zeros(epoch)
training_loss = np.zeros(epoch)

for i in range(epoch):
    with tf.GradientTape() as tape:
        predicted_parameter_mat = train_network(z)
        loss = gt.log_ising_null(x_y_mat, predicted_parameter_mat)
    grads = tape.gradient(loss, train_network.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

    training_loss[i] = loss.numpy()
    kl_divergence[i] = gt.kl_divergence(true_parameter_mat, predicted_parameter_mat)
    if i % 10 == 0:
        print("Iteration %d, the loss is %f " % (i, loss))
        print("The KL divergence is %f" % kl_divergence[i])
