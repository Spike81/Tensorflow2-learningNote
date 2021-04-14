import numpy as np

# read the .csv file
def readFile(filePath):
    point = np.genfromtxt(filePath, delimiter=",")
    return point

def compute_loss(point, w, b):
    totalLoss = 0
    for i in range(0, len(point)):
        x = point[i, 0]
        y = point[i, 1]
        totalLoss += (w * x + b - y) ** 2
        pass
    return totalLoss/float(len(point))

def update(point, w, b, lr):
    # compute the grad
    w_grad = 0
    b_grad = 0
    for i in range(0, len(point)):
        x = point[i, 0]
        y = point[i, 1]
        w_grad += 2 * (w * x + b - y) * x * 1/float(len(point))
        b_grad += 2 * (w * x + b - y) * 1/float(len(point))
        pass
    new_w = w - lr * w_grad
    new_b = b - lr * b_grad
    return [new_w, new_b]


def run():
    point = readFile("E:\Deep Learning\Deep-Learning-with-TensorFlow-book\ch02-回归问题\data.csv")
    # print(point)
    # find the loss
    w = 0
    b = 0
    loss = compute_loss(point, w, b)
    print("the start loss: %f" %loss)
    # min the loss
    num_iteration = 1000
    lr = 0.0001
    for i in range(num_iteration):
        w, b = update(point, w, b, lr)
        new_loss = compute_loss(point, w, b)
        print(new_loss)
        pass
    y = w * 53.4268 + b
    print("y = %f" %y)


if __name__ == "__main__":
    run()
    pass
