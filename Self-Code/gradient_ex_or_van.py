# !/usr/local/bin/python3
# @Time : 2021/4/15 19:35
# @Author : Tianlei.Shi
# @Site :
# @File : gradient_ex_or_van.py
# @Software : PyCharm

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)

from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import matplotlib.pyplot as plt

(x_, y_), _ = datasets.mnist.load_data()
x_ = tf.convert_to_tensor(x_, dtype=tf.float32) / 255.
y_ = tf.convert_to_tensor(y_, dtype=tf.int32)
y_ = tf.one_hot(y_, depth=10)

train_db = tf.data.Dataset.from_tensor_slices((x_,y_)).batch(128).repeat(30)
x,y = next(iter(train_db))

if __name__ == '__main__':

    w1, b1 = tf.Variable(tf.random.truncated_normal([784,512], stddev=0.1)), tf.Variable(tf.zeros([512]))
    w2, b2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    optimizer = optimizers.SGD(lr=1)

    for step, (x, y) in enumerate(train_db):

        x = tf.reshape(x, (-1, 28*28))

        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            out = h2 @ w3 + b3

            loss = tf.reduce_mean(tf.square(out - y), axis=1)

        grads = tape.gradient(loss, [w1,b1,w2,b2,w3,b3])

        print("==before==")
        for g in grads:
            print(tf.norm(g))

        grads, _ = tf.clip_by_global_norm(grads, 15)

        print("==after==")
        for g in grads:
            print(tf.norm(g))

        optimizer.apply_gradients(zip(grads, [w1,b1,w2,b2,w3,b3]))