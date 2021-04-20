# !/usr/local/bin/python3
# @Time : 2021/4/20 20:10
# @Author : Tianlei.Shi
# @Site :
# @File : fashionMinist.py
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
from tensorflow.keras import layers, datasets, optimizers, Sequential, metrics

def process(x, y):
    '''
    preprocess data
    :return: data
    '''

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def iter_db(db):
    db = iter(db)
    simple = next(db)
    print(simple[0].shape, simple[1].shape)

# [60k, 28, 28] [60k], [10k, 28, 28] [20k]
(x, y), (x_, y_) = datasets.fashion_mnist.load_data()
# print(x.shape, y.shape, x_.shape, y_.shape)
# print(x.max(), x.min())
# print(y.max(), y.min())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(process).shuffle(10000).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_, y_))
test_db = test_db.map(process).batch(128)

iter_db(train_db)

model = Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10)
])

model.build(input_shape=[None, 28*28])
model.summary()
optimizers = optimizers.Adam(lr=0.001)

for epoch in range(10):

    for step, (x, y) in enumerate(train_db):

        x = tf.reshape(x, [-1, 28*28])
        y = tf.one_hot(y, depth=10)

        with tf.GradientTape() as tape:
            logits = model(x)
            loss_mse = tf.reduce_mean(tf.losses.MSE(y, logits))
            loss_cs = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))

        grads = tape.gradient(loss_cs, model.trainable_variables)
        optimizers.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, "loss: ", float(loss_mse), float(loss_cs))


    correctNum = 0
    num = 0

    for x, y in test_db:
        x = tf.reshape(x, [-1, 28*28])

        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        prob = tf.argmax(prob, axis=1)
        prob = tf.cast(prob, dtype=tf.int32)

        correct = tf.equal(prob, y)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

        correctNum += correct
        num += x.shape[0]

    acc = correctNum / num
    print("epoch:", epoch, "acc:", float(acc))







