# !/usr/local/bin/python3
# @Time : 2021/4/22 21:39
# @Author : Tianlei.Shi
# @Site :
# @File : saved_model.py
# @Software : PyCharm

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, datasets
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

network.fit(db, epochs=1, validation_data=ds_val, validation_freq=2)

network.evaluate(ds_val)

tf.saved_model.save(network, "saved_model/")
print('saved total model.')
del network

network = tf.saved_model.load("saved_model/")
# f = network.signatures["serving_default"]
# print(f(x=tf.constant([[1.]])))
# # print("fsdfdsf")
pruned = network.prune("x:0", "out:0")
print(pruned(tf.ones([])))