import tensorflow as tf
import numpy as np

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=2,
            input_shape=(3,),
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output

    def dummy_r un(self):
        """
        This method is to let python initialize the network and weights not just the computation graph.
        :return: None.
        """

        dummy_z = np.random.normal(0, 1, (3, 5))
        self(dummy_z)


model = Linear()
model.dummy_run()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model(X)

model.dense.set_weights( [ np.array( [2,3,4] ).reshape(3, 1), np.array([3]) ]   )

model.dense.get_weights()



mse = tf.keras.losses.MeanSquaredError()

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



tf.debugging.set_log_device_placement(True)

