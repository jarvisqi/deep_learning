# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.power(x, 2)


def d_f_1(x):
    return 2.0*x


def d_f_2(f, x, delta=1e-4):
    return (f(x+delta) - f(x-delta)) / (2*delta)


def gradient_descent():
    """梯度下降求最小值
    """

    data = np.arange(-10, 11)
    plt.plot(data, f(data))
    plt.show()

    learning_rate = 0.1
    epochs = 30

    x_init = 10.0
    x = x_init
    for i in range(epochs):
        d_f_x = d_f_1(x)
        # d_f_x = d_f_2(f, x)
        x = x - learning_rate * d_f_x
        print(x)

    print('initial x =', x_init)
    print('arg min f(x) of x =', x)
    print('f(x) =', f(x))


if __name__ == '__main__':
    gradient_descent()
