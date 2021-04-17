# !/usr/local/bin/python3
# @Time : 2021/4/17 17:17
# @Author : Tianlei.Shi
# @Site :
# @File : mlp.py
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

from tensorflow import keras

x = tf.random.normal([2,3])

model = keras.Sequential([
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(2)
])

model.build(input_shape=[None, 3])  # 这里的 3，代表 w.shape[0]
model.summary()

for p in model.trainable_variables:
    print(p.name, p.shape)