# !/usr/local/bin/python3
# @Time : 2021/4/11 17:29
# @Author : Tianlei.Shi
# @Site :
# @File : forward.py
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
from tensorflow.keras import datasets

(x, y), _ = datasets.mnist.load_data()  # ([60k, 28, 28], [60k])
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

# print(x.shape, y.shape)
# print(tf.reduce_min(x), tf.reduce_max(x), tf.reduce_min(y), tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
# print("batch:", sample[0].shape, sample[1].shape)  # batch: (128, 28, 28) (128,)

# [b, 784] => ... => [b, 10]
# w.shape = [dim_in, dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-4

for epoch in range(10):

    for step, (x, y) in enumerate(train_db):

        # print(x.shape)

        # for (x, y) in train_db:
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as tape:
            # h1 = w1 @ x + b1
            h1 = tf.nn.relu(x @ w1 + b1)

            h2 = tf.nn.relu(h1 @ w2 + b2)

            out = h2 @ w3 + b3

            # print(out.shape)

            # loss
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_onehot - out))

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        w1.assign_sub(lr * grads[0])  # w1 = w1 - lr * grads[0]
        b1.assign_sub(lr * grads[1])

        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print("epoch", epoch, "step", step, "loss:", float(loss))