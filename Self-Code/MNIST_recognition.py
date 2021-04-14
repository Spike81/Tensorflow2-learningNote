# !/usr/local/bin/python3
# @Time : 2021/3/7 0:20
# @Author : Tianlei.Shi
# @Site :
# @File : MNIST_recognition.py
# @Software : PyCharm

'''
1. pre-process:
    1. load data
    2. convert to tensor (pic become gray-pic, label become one-hot type)
    3. structure a dataset (make connection between pic and label), and set the train batch

2. build model:
    1. structure model (layer, activation)
    2. set learning rate

3. train:
    1. train one batch, then train one epoch
'''

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, Sequential, layers, optimizers

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()  # ((60000, 28, 28), (60000,)), ((10000, 28, 28), (10000,))
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.  # / 255 -> means become to gray-pic
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train, depth=10)  # depth is the category num

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)

model = Sequential([layers.Dense(512, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)  # update parameters w and b automatically

for i in range(30):

    for step, (x, y) in enumerate(train_dataset):  # loop 300 times, one batch (200 pic) per time
        # print(step, x.shape, y.shape)

        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28*28))  # make it one-dis
            out = model(x)  # get result

            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]  # calculate loss, x.shape[0] = 200

        grads = tape.gradient(loss, model.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(step, "loss:", loss.numpy())