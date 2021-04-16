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

y = tf.linspace(-2, 2, 5)  # 在 -2 和 2 直接进行采样，采样 5 个点
x = tf.linspace(-2, 2, 5)
point_x, point_y = tf.meshgrid(x,y)
print(point_y.shape)
