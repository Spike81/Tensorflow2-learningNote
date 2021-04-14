import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_label), _ = datasets.mnist.load_data() # x_train: [60k,28,28], y:[60k]

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)/255
y_label = tf.convert_to_tensor(y_label, dtype=tf.int32)
# print(x_train.shape, y_label.shape, x_train.dtype, y_label.dtype)
# print(tf.reduce_min(x_train), tf.reduce_max(x_train))
# print(tf.reduce_min(y_label), tf.reduce_max(y_label))

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_label)).batch(128)
train_iter = iter(train_db)
simple = next(train_iter)
# print("batch:", simple[0].shape, simple[1].shape)

# [b,784] => [b, 256] => [b, 128] => [b, 10]
#  [dim_in, dim_out], [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3, b3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

lr = 0.0051 # 0.001

for epoch in range(10):

    for step, (x_train, y_label) in enumerate(train_db):
        x = tf.reshape(x_train, [-1, 28*28])

        with tf.GradientTape() as tape:
            # h1 = x@w1 + b
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3

            # compute loss
            y_onehot = tf.one_hot(y_label, depth=10)

            # mse = mean(sum(y-out)^2)
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)
            pass

        # compute gradients
        grads = tape.gradient(loss, [w1,b1,w2,b2,w3,b3])
        # w1 -= lr * w1_grad, update w1 ...
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, "loss:", float(loss))