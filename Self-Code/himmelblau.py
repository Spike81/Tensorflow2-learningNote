# !/usr/local/bin/python3
# @Time : 2021/4/20 19:40
# @Author : Tianlei.Shi
# @Site :
# @File : himmelblau.py
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
import matplotlib.pyplot as plt

# plot

def himmelblau(x):
    '''
    the himmelblau function
    :param x: x factor
    :return: the y
    '''

    return (x[0] ** 2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 -7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)

X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

fig = plt.figure("himmelblau")
ax = fig.gca(projection="3d")
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

input = tf.constant([0., 0.])

lr = 0.01

for i in range(200):

    with tf.GradientTape() as tape:
        tape.watch([input])
        z = himmelblau(input)

    grad = tape.gradient(z, input)
    # print(grad, type(grad[0]))
    input -= 0.01 * grad

print(himmelblau(input))