# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

"""
Tensorboard： 可视化逻辑回归训练过程
"""


def main():
    
    # 1. numpy造数据
    X = np.linspace(-1, 1, 100)[:, np.newaxis]
    noise = np.random.normal(0, 0.1, size=X.shape)
    y = np.power(X, 2)+noise

    # 2. 定义tensor 变量    
    with tf.variable_scope("Inputs"):
        input_X = tf.placeholder(tf.float32, X.shape, name="input_X")
        input_y = tf.placeholder(tf.float32, y.shape, name="input_y")

    # 3. 定义神经网络结构input + 1hidden(with RELU激活函数) + ouput
    with tf.variable_scope("Net"):
        hidden_1 = tf.layers.dense(input_X, 10, activation=tf.nn.relu, name="hidden_layer_1")
        output = tf.layers.dense(hidden_1, 1, name="output_layer")

        # 添加到Tensorboard直方图
        tf.summary.histogram("hidden_out",hidden_1)
        tf.summary.histogram("prediction",output)

    # 4. 应用mean square 计算cost损失值
    loss=tf.losses.mean_squared_error(input_y,output,scope="loss")
    # 5. 应用梯度下降优化器优化loss
    train_op=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
    tf.summary.scalar("loss",loss)     # 将loss添加到Tensorboard标量

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./logs", graph=sess.graph)   # 写Buffer到文件
    merge_op = tf.summary.merge_all()

    for step in range(100):

        _, result = sess.run([train_op, merge_op], {input_X: X, input_y: y})
        writer.add_summary(result, step)



if __name__ == '__main__':
    main()
