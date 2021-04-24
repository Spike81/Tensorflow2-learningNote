# !/usr/local/bin/python3
# @Time : 2021/4/24 22:03
# @Author : Tianlei.Shi
# @Site :
# @File : cifar10.py
# @Software : PyCharm

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers, datasets

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        # self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x


class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32 * 32 * 3, 256)
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



def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
y_train = tf.squeeze(y_train)
y_val = tf.squeeze(y_val)

y_train = tf.one_hot(y_train, depth=10)
y_val = tf.one_hot(y_val, depth=10)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(128)

sample = next(iter(train_db))
print("batch:", sample[0].shape, sample[1].shape)

model = MyNetwork()
model.build(input_shape=(None, 32*32*3))
model.summary()

model.compile(optimizer=optimizers.Adam(lr=0.001),
                      loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
model.fit(train_db, epochs=10, validation_data=test_db, validation_freq=2)

model.evaluate(test_db)

model.save_weights("weight/weights.ckpt")
del model

model = MyNetwork()
model.compile(optimizer=optimizers.Adam(lr=0.001),
                      loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

# 加载 model 前，先要使用 Sequential 创建 model 结构
model.load_weights("weight/weights.ckpt")
model.evaluate(test_db)