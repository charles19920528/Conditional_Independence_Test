import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import generate_train_fucntions as gt
from generate_z import dim_z, sample_size_vet

hidden_1_out_dim = 3

# Create null network
np.random.seed(1)
null_network_generate = gt.IsingNetwork(dim_z, hidden_1_out_dim, 3)
null_network_generate.dummy_run()

linear_1_weight_array = np.random.normal(1, 1, size=(dim_z, hidden_1_out_dim))
linear_1_bias_array = np.zeros(shape=(hidden_1_out_dim,))

linear_2_weight_array = np.random.normal(1, 1, size=(hidden_1_out_dim, 3))
linear_2_bias_array = np.zeros(shape=(3,))

null_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])



class IsingSimulation:
    def __init__(self, z_mat, true_network, null_boolean,learning_rate, buffer_size, batch_size, epoch):
        """
        Create a class which can generate data and train a network.
        :param z_mat: A n by p dimension numpy array / tensor. n is the sample size. This is the data we condition on.
        :param true_network: An object of IsingNetwork class. This is the true network we use to generate parameters.
        :param learning_rate: A scalar which is a (hyper)parameter in the tf.keras.optimizers.Adam function.
        :param buffer_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.shuffle function.
        :param batch_size: A scalar which is a (hyper)parameter in the tf.data.Dataset.batch function.
        :param epoch: A scalar indicating the number of times training process pass through the data set.
        """
        self.z_mat = z_mat
        self.sample_size = z_mat.shape[0]
        self.true_network = true_network
        self.null_boolean = null_boolean
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch = epoch

    def generate_x_y_mat(self):
        """
        The function will generate the matrix of responses (x, y).
        :param null_boolean: A boolean value indicating if we are simulating under the the independence assumption.
        :return: x_y_mat
        """
        true_parameter_mat = self.true_network(self.z_mat)
        if self.null_boolean:
            p_equal_1_mat = gt.pmf_null(1, true_parameter_mat)
            x_y_mat = np.random.binomial(n=1, p=p_equal_1_mat, size=(self.sample_size, 2)).astype("float32") * 2 - 1

            return x_y_mat
        else:
            p_mat = gt.pmf_collection(true_parameter_mat)
            # Recall that the column of p_mat corresponds to P(X = 1, Y = 1),
            # P(X = 1, Y = -1), P(X = -1, Y = 1) and P(X = -1, Y = -1)
            raw_sample_mat = np.zeros((self.sample_size, 4))
            for i in np.arange(self.sample_size):
                p_vet = p_mat[i, :]
                raw_sample = np.random.multinomial(1, p_vet)
                raw_sample_mat[i, :] = raw_sample
            conversion_mat = np.array([
                [1, 1], [1, -1], [-1, 1], [-1, -1]
            ])
            x_y_mat = raw_sample_mat.dot(conversion_mat)

            return x_y_mat

    def trainning(self, x_y_mat):
        pass


z_mat = np.loadtxt("./data/z_%d.txt" % 30, dtype="float32")
issim = IsingSimulation(z_mat = z_mat, true_network = null_network_generate, null_boolean = False,learning_rate = 0.005,
                        buffer_size = 1000, batch_size = 50, epoch = 2)



parameter_mat = tf.constant([[1,2,3], [4,5, 6]], dtype = "float32")
p_equal_1_mat = gt.pmf_collection(parameter_mat)
test = np.random.multinomial(1, p_equal_1_mat[1, :])


p_equal_1_mat = np.array([1,0, 0, 1]).reshape(2, 2)





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
    loss_kl_array = np.zeros((2, epoch * len(list(train_ds))))
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
