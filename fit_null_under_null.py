import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import evaluation_functions as ef
import pickle
from generate_z import dim_z, sample_size_vet

####################
# Hyper parameters #
####################
hidden_1_out_dim = 3

######################
# Fit the null model #
######################
# Create null network
np.random.seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
null_network_generate.dummy_run()

linear_1_weight_array = np.random.normal(1, 1, size=(dim_z, hidden_1_out_dim))
linear_1_bias_array = np.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = np.random.normal(1, 1, size=(hidden_1_out_dim, 2))
linear_2_bias_array = np.zeros(shape=(2,))

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])

# See results for just one sample.
loss_one_sample_dictionary = dict()

for sample_size in sample_size_vet:
    # Produce raw data
    z_mat = np.loadtxt("./data/z_%d.txt" % sample_size, dtype="float32")
    true_parameter_mat = null_network_generate(z_mat)
    p_equal_1_mat = gt.pmf_null(1, true_parameter_mat)
    x_y_mat = np.random.binomial(n=1, p=p_equal_1_mat, size=(sample_size, 2)).astype("float32") * 2 - 1

    train_network = gt.IsingNetwork(dim_z, hidden_1_out_dim, 2)
    train_network.dummy_run()

    # Hyperparameters for training.
    learning_rate = 0.005
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    buffer_size = 1000
    batch_size = 50
    epoch = 1000

    # Prepare training data.
    train_ds = tf.data.Dataset.from_tensor_slices((z_mat, x_y_mat))
    train_ds = train_ds.shuffle(buffer_size).batch(batch_size)

    # Assume the first row corresponds to the likelihood
    loss_kl_array = np.zeros((2, epoch * len( list(train_ds) ) ) )
    loss_kl_array_index = 0
#    kl_divergence = np.zeros()
#    training_loss = np.zeros(epoch * len(list(train_ds)))

    for i in range(epoch):
        for z_batch, x_y_batch in train_ds:
            with tf.GradientTape() as tape:
                batch_predicted_parameter_mat = train_network(z_batch)
                loss = gt.log_ising_null(x_y_batch, batch_predicted_parameter_mat)
            grads = tape.gradient(loss, train_network.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, train_network.variables))

            predicted_parameter_mat = train_network(z_mat)

            loss_kl_array[0, loss_kl_array_index] = loss.numpy()
            loss_kl_array[1, loss_kl_array_index] = gt.kl_divergence(true_parameter_mat, predicted_parameter_mat)
            loss_kl_array_index += 1

        if i % 5 == 0:
            print("Sample size %d, Epoch %d" % (sample_size, i))
            print("The loss is %f " % loss)
            print("The KL divergence is %f" % loss_kl_array[1, loss_kl_array_index - 1])

    loss_one_sample_dictionary[sample_size] = loss_kl_array

with open("./results/loss_one_sample_dictionary.p", "wb") as fp:
    pickle.dump(loss_one_sample_dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


with open('./results/loss_one_sample_dictionary.p', 'rb') as fp:
    loss_one_sample_dictionary = pickle.load(fp)

loss_dict_null = ef.Loss_Dict(loss_one_sample_dictionary, sample_size_vet, 50)
loss_dict_null.plot_epoch_loss(100, 500)
loss_dict_null.plot_epoch_kl(200, 400)


"""
sample_size = 100
plt.figure(sample_size)
plt.plot(loss_one_sample_dictionary[sample_size][0,:], label = "likelihood")
plt.plot(loss_one_sample_dictionary[sample_size][1,:], label = "kl")
plt.legend()
plt.show()
plt.savefig("./figure/Loss function_%d.png" % sample_size)
"""



