# !/usr/local/bin/python3
# @Time : 2021/3/1 14:49
# @Author : Tianlei.Shi
# @Site :
# @File : linear_regression.py
# @Software : PyCharm

'''
1, load data
2, calculate loss
3, update parameters according to gradient
4, repeat a number of times
5, calculate final loss
'''

import numpy as np

def update_parameters(data, w, b, lr):
    gradient_w = 0
    gradient_b = 0
    N = float(len(data))
    for i in range(len(data)):
        gradient_w += (2 / N) * (w * data[i, 0] + b - data[i, 1]) * data[i, 0]
        gradient_b += (2 / N) * (w * data[i, 0] + b - data[i, 1])
    new_w = w - lr * gradient_w
    new_b = b - lr * gradient_b
    return new_w, new_b

def loss_calculater(data, init_w, init_b):
    loss = 0
    for i in range(len(data)):
        loss += (init_w * data[i, 0] + init_b - data[i, 1]) ** 2
    return loss / len(data)

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=",")
    return data

if __name__ == '__main__':
    file_path = "E:\\Deep Learning\\Deep-Learning-with-TensorFlow-book\\ch02-回归问题\\data.csv"
    data = load_data(file_path)
    # print(data)
    init_w = 0
    init_b = 0
    init_loss = loss_calculater(data, init_w, init_b)
    print(init_loss)

    lr = 0.0001
    w = init_w
    b = init_b
    iter_num = 1000
    for i in range(iter_num):
        w, b = update_parameters(data, w, b, lr)

    loss = loss_calculater(data, w, b)
    print(loss, w, b)


# 5565.107834483211
# 112.61481011613473 1.4777440851894448 0.08893651993741344
