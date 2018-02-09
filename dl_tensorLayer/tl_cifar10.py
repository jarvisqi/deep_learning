import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from skimage import io, transform
import tensorflow as tf
import tensorlayer as tl
import numpy as np


X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y = tf.placeholder(tf.int64, shape=[None, ], name='y')

def train():

    sess = tf.InteractiveSession()
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

    net = build_net(X)
    outputs = net.outputs

    # 定义损失函数和优化器
    loss = tl.cost.cross_entropy(outputs, y, name='loss')
    L2 = 0
    for p in tl.layers.get_variables_with_name('relu/W', True, True):
        L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
    loss = loss + L2

    correct_prediction = tf.equal(tf.argmax(outputs, 1), y)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义优化器
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001).minimize(loss, var_list=net.all_params)
    
    tf.summary.scalar('acc', acc)
    tl.layers.initialize_global_variables(sess)

    # 打印网络信息
    net.print_layers()
    net.print_params(False)

    # 训练模型
    tl.utils.fit(sess, net, optimizer, loss, X_train, y_train, X, y,
                 acc=acc, batch_size=128, n_epoch=32, print_freq=1, X_val=X_test, y_val=y_test,
                 tensorboard=True,tensorboard_epoch_freq=1)

    # 把模型保存成 .npz 文件
    tl.files.save_npz(net.all_params, name='./models/cifar10_model.npz')
    sess.close()


def build_net(x):
    """
    定义网络结构
    """
    # 定义模型
    print("build network")
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)

    net = tl.layers.InputLayer(x, name='input_layer')

    net = tl.layers.Conv2d(net, 64, (5, 5), act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv1')
    net = tl.layers.MaxPool2d(net, (3, 3), padding='SAME', name='maxpool1')
    net = tl.layers.BatchNormLayer(net, name='nonrm1')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop1')

    net = tl.layers.Conv2d(net, 128, (5, 5), act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv2')
    net = tl.layers.MaxPool2d(net, (3, 3), padding='SAME', name='maxpool2')
    net = tl.layers.BatchNormLayer(net, name='nonrm2')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')

    # net = tl.layers.Conv2d(net, 256, (5, 5), act=tf.nn.relu, name='conv3')
    # net = tl.layers.MaxPool2d(net, (3, 3), padding='SAME', name='maxpool3')
    # net = tl.layers.BatchNormLayer(net, name='nonrm3')
    # net = tl.layers.DropoutLayer(net, keep=0.5, name='drop3')

    net = tl.layers.FlattenLayer(net, name='flatten')
    net = tl.layers.DenseLayer(net, n_units=512, act=tf.nn.relu,W_init=W_init2, b_init=b_init2, name='dense1')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop4')
    net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity,W_init=tf.truncated_normal_initializer(stddev=1/192.0), name='output')

    return net



if __name__ == '__main__':
    train()
