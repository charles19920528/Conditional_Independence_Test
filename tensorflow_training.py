import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

###################
# Data Generation #
###################
np.random.seed(1)
sample_size = 100
x = np.random.normal(0, 10, (sample_size, 1))
true_y = 3 * x - 2
y = true_y + np.random.normal(0, 5, (sample_size, 1))

plt.scatter(x, y)


sample_size = 10000
x = np.random.normal(0, 10, (sample_size, 2))
beta_mat = np.array([2, -5]).reshape(2, 1)
true_y = x.dot(beta_mat) + 3
y = true_y + np.random.normal(0, 5, (sample_size, 1))


###############
#
###############
class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.linear = tf.keras.layers.Dense(
            input_dim = 2,
            units = 1,
            activation = None,
            kernel_initializer = tf.initializers.RandomUniform(-1, 1),
            bias_initializer = tf.initializers.RandomUniform(-1, 1)
        )

    def call(self, input):
        output = self.linear(input)
        return output


def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))


model = LinearModel()
model(x)
#model.set_weights([ np.array([2, -5]).reshape(2, 1), np.array([3])])
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
for i in range(1000):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = mse_loss(y_pred, y)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
    if i % 10 == 0:
        print("The loss is %f." % loss)

