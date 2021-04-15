# !/usr/local/bin/python3
# @Time : 2021/3/8 13:57
# @Author : Tianlei.Shi
# @Site :
# @File : tensor.py
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

import numpy as np
from math import e

indices = tf.constant([[4],[3],[1],[7]])
updates = tf.constant([9,8,11,12])
shape = tf.constant([8])

a = tf.scatter_nd(indices, updates, shape)
print(a)  # tf.Tensor([ 0 11  0  8  9  0  0 12], shape=(8,), dtype=int32)
