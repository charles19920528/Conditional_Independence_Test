import numpy as np
import tensorflow as tf
import generate_train_fucntions as gt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Generate the data using the alternative but fit the alternative model.
alternative_net = gt.IsingNetwork(3, 3, 3)
z = np.loadtxt("./data/z_%d.txt" % 30, dtype="float32")
x_y_mat = np.loadtxt("./data/x_y_mat_%d.txt" % 30, dtype="float32")



