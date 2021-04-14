import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, optimizers, Sequential, metrics

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape)

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    return x,y

db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db = db.map(preprocess).shuffle(10000).batch(128)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(128)

db_iter = iter(db)
sample = next(db_iter)
print("batch:", sample[0].shape, sample[1].shape)

model = Sequential([
        layers.Dense(256, activation=tf.nn.relu), # [b, 784] => [b, 256]
        layers.Dense(128, activation=tf.nn.relu), # [b, 256] => [b, 128]
        layers.Dense(64, activation=tf.nn.relu), # [b, 128] => [b, 64]
        layers.Dense(32, activation=tf.nn.relu), # [b, 64] => [b, 32]
        layers.Dense(10) # [b, 32] => [b, 10] param # = 32 * 10 + 10 = 330

])
model.build(input_shape = [None, 28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)

def main():

    for epoch in range(30):

        for step, (x, y) in enumerate(db):

            # x = [b, 28, 28]
            # y = [b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:

                logits = model(x) # [b, 784] => [b, 10]
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
                pass

            grad = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, "loss: ", float(loss_mse), float(loss_ce))
                pass


        # test

        correct_total = 0
        correct_num = 0

        for x, y in db:
            x = tf.reshape(x, [-1, 28*28])

            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            prob = tf.argmax(prob, axis=1) # prob: [b]
            prob = tf.cast(prob, dtype=tf.int32)

            correct = tf.equal(prob, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            correct_total = int(correct)
            correct_num = x.shape[0]

            pass

        acc = correct_total / correct_num
        print(epoch, "acc: ", acc)




    pass


if __name__ == '__main__':
    main()