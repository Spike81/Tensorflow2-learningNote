# !/usr/local/bin/python3
# @Time : 2021/5/10 14:44
# @Author : Tianlei.Shi
# @Site :
# @File : cifar100.py
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

tf.random.set_seed(2345)

conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

next(iter(train_db))

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

def main():
    conv_net = Sequential(conv_layers)
    # conv_net.build(input_shape=[None, 32, 32, 3])
    # x = tf.random.normal([4, 32, 32, 3])
    # out = conv_net(x)
    #
    # print(out.shape)

    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=tf.nn.relu),
    ])

    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])

    variables = conv_net.trainable_variables + fc_net.trainable_variables

    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(50):

        for step, (x, y) in enumerate(train_db):
            # print(x.shape)

            with tf.GradientTape() as tape:

                out = conv_net(x)

                out = tf.reshape(out, [-1, 512])

                logits = fc_net(out)

                y_onehot = tf.one_hot(y, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)

                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, loss)

        tt_num = 0
        tt_correct = 0
        for x, y in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            tt_num += x.shape[0]
            tt_correct += int(correct)

        acc = tt_correct / tt_num
        print(acc)



if __name__ == '__main__':
    main()