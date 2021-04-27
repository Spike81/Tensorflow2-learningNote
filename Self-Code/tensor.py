# !/usr/local/bin/python3
# @Time : 2021/3/8 13:57
# @Author : Tianlei.Shi
# @Site :
# @File : tensor.py
# @Software : PyCharm

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, datasets
from tensorflow import keras
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)

import numpy as np
from math import e

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()  # ((60000, 28, 28), (60000,)), ((10000, 28, 28), (10000,))
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.  # / 255 -> means become to gray-pic
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train, depth=10)  # depth is the category num

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)

model = Sequential([layers.Dense(512, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)  # update parameters w and b automatically

for i in range(1):

    for step, (x, y) in enumerate(train_dataset):  # loop 300 times, one batch (200 pic) per time
        # print(step, x.shape, y.shape)

        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28*28))  # make it one-dis
            out = model(x)  # get result

            loss = tf.reduce_sum(tf.losses.categorical_crossentropy(y, out, from_logits=True))  # calculate loss, x.shape[0] = 200

            loss_regularization = []

            for p in model.trainable_variables:
                loss_regularization.append(tf.nn.l2_loss(p))
            # print(loss_regularization)
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))

            loss = loss + 0.0001 * loss_regularization


        grads = tape.gradient(loss, model.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(step, "loss:", loss.numpy())