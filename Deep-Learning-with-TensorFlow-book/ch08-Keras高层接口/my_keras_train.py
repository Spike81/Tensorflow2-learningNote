import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y

batchsz = 128
# [32, 32, 3]
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.squeeze(y) # shape of y that input is [10k, 1]
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10) # [50k, 10]
y_val = tf.one_hot(y_val, depth=10) # [10k, 10]
print("datasets:", x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())

# setup a database
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print("batch:", sample[0].shape, sample[1].shape)


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_weight("w", [inp_dim, outp_dim])
        # self.bias = self.add_variable("b", [outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x

class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 256)
        self.fc3 = MyDense(256, 256)
        self.fc4 = MyDense(256, 256)
        self.fc5 = MyDense(256, 10)

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, [-1, 32*32*3])
        x = self.fc1(x)
        x = tf.nn.relu(x)

        x = self.fc2(x)
        x = tf.nn.relu(x)

        x = self.fc3(x)
        x = tf.nn.relu(x)

        x = self.fc4(x)
        x = tf.nn.relu(x)

        x = self.fc5(x)

        return x

network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)

network.evaluate(test_db)
network.save_weights("ckpt/weights.ckpt")
del network
print("save to ckpt/weights.ckpt")

network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
network.load_weights("ckpt/weights.ckpt")
print("loaded weights from file")
network.evaluate(test_db)