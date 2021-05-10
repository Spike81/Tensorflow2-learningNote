# !/usr/local/bin/python3
# @Time : 2021/5/10 20:57
# @Author : Tianlei.Shi
# @Site :
# @File : ResNet18_train.py
# @Software : PyCharm


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, datasets
from tensorflow import keras
from ResNet import ResNet18

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)

tf.random.set_seed(2345)


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
    model = ResNet18()

    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(50):

        for step, (x, y) in enumerate(train_db):
            # print(x.shape)

            with tf.GradientTape() as tape:

                logits = model(x)

                y_onehot = tf.one_hot(y, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)

                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, loss)

        tt_num = 0
        tt_correct = 0
        for x, y in test_db:
            logits = model(x)

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