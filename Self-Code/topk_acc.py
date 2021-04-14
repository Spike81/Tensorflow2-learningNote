# !/usr/local/bin/python3
# @Time : 2021/4/13 20:04
# @Author : Tianlei.Shi
# @Site :
# @File : topk_acc.py
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



def accuracy(output, target, topk=(1,)):
    topk = max(topk)
    acc = []

    for i in range(topk):
        pre = tf.math.top_k(output, topk).indices
        # print(pre)
        pre = tf.squeeze(tf.gather(pre, axis=1, indices=[i]))
        print(pre)
        compare = tf.reduce_sum(tf.cast(tf.equal(pre, target), dtype=tf.int32))
        if (acc == []):
            acc.append(compare.numpy() / output.shape[0])
        else:
            acc.append(((compare.numpy())/ output.shape[0])+acc[i-1])
    return acc



if __name__ == '__main__':
    output = tf.random.normal([10, 6])
    output = tf.math.softmax(output, axis=1)

    target = tf.random.uniform([10], maxval=6, dtype=tf.int32)

    print('prob:', output.numpy())
    print('label:', target.numpy())

    acc = accuracy(output, target, topk=(1,2,3,4,5,6))
    print('top-1-6 acc:', acc)