import numpy as np
import generate_train_fucntions as gt
import tensorflow as tf
from sklearn.datasets.samples_generator import make_blob
import hyperparameters as hp

seed_index = 1
tf.random.set_seed(seed_index)
np.random.seed(seed_index)

##################################
# Create the alternative network #
##################################
alt_network_generate = gt.IsingNetwork(hp.dim_z, hp.hidden_1_out_dim, 3)
alt_network_generate.dummy_run()

linear_1_weight_array = tf.random.normal(shape = (hp.dim_z, hp.hidden_1_out_dim), mean = 1, stddev = 1)
linear_1_bias_array = tf.zeros(shape = (hp.hidden_1_out_dim, ))

linear_2_weight_array = tf.random.normal(shape = (hp.hidden_1_out_dim, 3), mean = 1, stddev = 0.1)
linear_2_bias_array = tf.zeros(shape=(3, ))

alt_network_generate.set_weights([linear_1_weight_array, linear_1_bias_array,
                                   linear_2_weight_array, linear_2_bias_array])
