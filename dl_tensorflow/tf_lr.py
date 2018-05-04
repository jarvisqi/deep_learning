# tensorflow 实现线性回归
# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def main():
    num_points = 100
    vectors_set = list()

    for i in range(num_points):
        x1 = np.random.normal(0.00, 00.55)
        y1 = x1*0.1+0.3+np.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

    plt.scatter(x_data, y_data, c='r')
    plt.show()

    W = tf.Variable(tf.random_uniform([1], -1, 1), name="W")
    b = tf.Variable(tf.zeros([1]), name="b")
    y = W*x_data+b

    # 定义损失函数
    loss = tf.reduce_mean(tf.square(y-y_data), name="loss")
    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss, name="train")

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # 初始化 W 和 b
    print("初始化值：W=", sess.run(W), "b=", sess.run(b))

    for step in range(20):
        sess.run(train)
        # 打印每次训练的 W和b
        print("第{}步：W={}, b={}".format(step, sess.run(W), sess.run(b)))

    # 展示
    plt.scatter(x_data, y_data, c="r")
    plt.plot(x_data, sess.run(W)*x_data+sess.run(b), c="b")
    plt.show()


if __name__ == '__main__':
    main()
