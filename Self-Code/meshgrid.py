# !/usr/local/bin/python3
# @Time : 2021/4/16 20:51
# @Author : Tianlei.Shi
# @Site :
# @File : meshgrid.py
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

import matplotlib.pyplot as plt

def func(point):
    z = tf.math.sin(point[..., 0]) + tf.math.sin(point[..., 1])
    return z


x = tf.linspace(0., 2*3.14, 500)
y = tf.linspace(0., 2*3.14, 500)

x, y = tf.meshgrid(x, y)

point = tf.stack([x, y], axis=2)
print(point.shape)

z = func(point)
print(z.shape)

plt.figure("plot 2d func value")
plt.imshow(z, origin="lower", interpolation="none")
plt.colorbar()

plt.figure("plot 2d func contour")
plt.contour(x, y, z)
plt.colorbar()
plt.show()