import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

def accuracy(output, target, topk=(1.)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1,0])
    target_ = tf.broadcast_to(target, pred.shape)

    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], -1), dtype=tf.int32)
        print("the accuary of top_{} is {}" .format(k, correct_k))
        correct_k = tf.reduce_sum(correct_k)
        acc = float(100 * correct_k / batch_size) # acc / 100 = correct / total => acc = 100 * correct / total
        res.append(acc)

    return res


tf.random.set_seed(2467)

output = tf.random.normal([10,6]) # 10 objects, 6 categories
output = tf.math.softmax(output, axis=1)

target = tf.random.uniform([10], maxval=6, dtype=tf.int32)

print("prob: ", output.numpy())

pred = tf.argmax(output, axis=1)
print("pred: ", pred.numpy())
print("label: ", target.numpy())

acc = accuracy(output, target, topk=(1,2,3,4,5,6))
print("acc is ", acc)
