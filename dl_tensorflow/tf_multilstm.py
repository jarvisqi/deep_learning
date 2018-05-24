import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.logging.set_verbosity(tf.logging.ERROR)


# 首先导入数据，看一下数据的形式
mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
print(mnist.train.images.shape)


# 设置超参数
lr = 1e-3
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
batch_size = tf.placeholder(tf.int32, name='batch_size')

n_batch_size = 128
# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
n_input = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
n_steps = 28
# 每个隐含层的节点数
n_hidden = 256
# LSTM layer 的层数
n_layers = 3
# 最后输出分类类别数量，如果是回归预测的话应该是 1
n_classes = 10



def main():
    # **步骤1：RNN 的输入shape = (n_batch_size, timestep_size, n_input)
    X = tf.placeholder(tf.float32, shape=(None, n_steps*n_input), name="X")
    y = tf.placeholder(tf.float32, shape=(None, n_classes), name="y")

    def lstm_cell(n_hidden,keep_prob): 
        # **步骤2：定义一层 LSTM_cell，只需要说明 n_hidden, 它会自动匹配输入的 X 的维度
        cell = rnn.LSTMCell(num_units=n_hidden,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2),
                            forget_bias=1.0,
                            state_is_tuple=True)
        # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
        cell = rnn.DropoutWrapper(cell=cell,
                                  input_keep_prob=1.0,
                                  output_keep_prob=keep_prob)
        return cell 
    
    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    cells = [lstm_cell(n_hidden,keep_prob) for _ in range(n_layers)]
    mlstm_cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
    # **步骤5：用全零来初始化state  # 通过zero_state得到一个全0的初始状态
    init_state = mlstm_cell.zero_state(n_batch_size, dtype=tf.float32)

    # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [n_batch_size, timestep_size, n_hidden]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, n_batch_size, n_hidden],
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [n_batch_size, n_hidden]
    # outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
    # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
    # 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
    # **步骤6：方法二，按时间步展开计算

    outputs = list()
    with tf.variable_scope('RNN'):
        for timestep in range(n_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态
            cell_output, h1 = mlstm_cell.call(X, init_state)
            outputs.append(cell_output)
    h_state = outputs[-1]

    # 上面 LSTM 部分的输出会是一个 [n_hidden] 的tensor，我们要分类的话，还需要接一个 softmax 层
    # 首先定义 softmax 的连接权重矩阵和偏置
    # out_W = tf.placeholder(tf.float32, [n_hidden, n_classes], name='out_Weights')
    # out_bias = tf.placeholder(tf.float32, [n_classes], name='out_bias')
    # 开始训练和测试

    W = tf.Variable(tf.truncated_normal([n_hidden, n_classes]), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[n_classes]), dtype=tf.float32)
    y_ = tf.nn.softmax(tf.matmul(h_state, W)+bias)

    # 损失和评估函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(2000):
            
            train_X, train_y = mnist.train.next_batch(n_batch_size)
            session.run(train_op, feed_dict={X: train_X,
                                             y: train_y,
                                             keep_prob: 0.5
                                             })
            if (i+1) % 200 == 0:
                train_acc, train_loss = session.run([acc, cost], feed_dict={X: train_X,
                                                                            y: train_y,
                                                                            keep_prob: 1.0
                                                                            })
                # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
                print("Iter {}, step {}, loss {:6f}, train acc {}".format(mnist.train.epochs_completed,
                                                                      (i+1),
                                                                      train_loss,
                                                                      train_acc))
        print("\nevaluation model")
        test_X = mnist.test.images[:n_batch_size]
        test_y = mnist.test.labels[:n_batch_size]
        # 计算测试数据的准确率
        test_acc, test_loss = session.run([acc, cost], feed_dict={
            X: test_X,
            y: test_y,
            keep_prob: 1.0,
            batch_size: n_batch_size
        })
        print("test acc {},test loss {}".format(test_acc, test_loss))


if __name__ == '__main__':
    main()


