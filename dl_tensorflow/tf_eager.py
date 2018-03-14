# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf 
import tensorflow.contrib.eager as tfe 
import numpy as np
import matplotlib.pyplot as plt

tfe.enable_eager_execution()

def main():

    print(tf.add(1, 2))
    print(tf.add([1, 2], [3, 4]))
    print(tf.square(5))
    print(tf.reduce_sum([1, 2, 3]))
    print(tf.encode_base64("hello world"))

    x = tf.constant(2)
    y = tf.constant(3)
    print(x * y + 1)

    print(tf.contrib.signal.hamming_window(x * y + 1))

    ones = np.ones([3, 3])
    print("numpy 3x3 matrix of 1s:")
    print(ones)
    print("")

    print("Multiplied by 42:")
    print(tf.multiply(ones, 42))


num_training_steps = 10


def train_model():
    true_w = 3
    true_b = 2
    NUM_EXAMPLES = 1000
    inputs = tf.random_normal(shape=[NUM_EXAMPLES, 1])
    noise = tf.random_normal(shape=[NUM_EXAMPLES, 1])
    labels = inputs * true_w + true_b + noise
    # 数据可视化
    plt.scatter(inputs.numpy(), labels.numpy())
    plt.show()

    wb = tf.layers.Dense(units=1, use_bias=True)
    # # 创建梯度函数
    value_and_gradients_fn = tfe.implicit_value_and_gradients(loss_fn)
    
    # # 计算梯度
    # loss, grads_and_vars = value_and_gradients_fn(inputs, labels, wb)
    # print('Loss: {}'.format(loss))
    # for (grad, var) in grads_and_vars:
    #     print("")
    #     print('Gradient: {}\nVariable: {}'.format(grad, var))

    # 创建优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    # print("w, b, 将求出的梯度应用到变量之前:")
    # w, b = wb.variables
    # print(w.read_value().numpy(), b.read_value().numpy())
    # print()
    # # 计算梯度
    # empirical_loss, gradients_and_variables = value_and_gradients_fn(inputs, labels, wb)
    # print('Loss: {}'.format(empirical_loss))
    # for (grad, var) in gradients_and_variables:
    #     print("")
    #     print('Gradient: {}\nVariable: {}'.format(grad, var))
    # optimizer.apply_gradients(gradients_and_variables)

    # print("w, b, 将求出的梯度应用到变量之后:")
    # w, b = wb.variables
    # print(w.read_value().numpy(), b.read_value().numpy())
    # print()

    loss_at_step = []
    w_at_step = []
    b_at_step = []

    print("\n训练")
    for step_num in range(num_training_steps):
        loss, gradients_and_variables = value_and_gradients_fn(inputs, labels, wb)
        print('Loss: {}'.format(loss))
        loss_at_step.append(np.asscalar(loss.numpy()))
        w, b = wb.variables
        print("之前：",w.read_value().numpy(), b.read_value().numpy())
        print()
        optimizer.apply_gradients(gradients_and_variables)
        w, b = wb.variables
        print("之后：",w.read_value().numpy(), b.read_value().numpy())
        print()
        w_at_step.append(np.asscalar(w.read_value().numpy()))
        b_at_step.append(np.asscalar(b.read_value().numpy()))

    print(w_at_step)
    t = range(0, num_training_steps)
    plt.plot(t, loss_at_step, 'k',
            t, w_at_step, 'r',
            t, [true_w] * num_training_steps, 'r--',
            t, b_at_step, 'b',
            t, [true_b] * num_training_steps, 'b--')
    plt.legend(['loss', 'w estimate', 'w true', 'b estimate', 'b true'])
    plt.show()


def loss_fn(inputs, labels, wb):
    predictions = wb(inputs)
    return tf.reduce_mean(tf.square(predictions - labels))


if __name__ == '__main__':
    # main()

    train_model()

    
