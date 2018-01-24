#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time, os, io
from PIL import Image

sess = tf.InteractiveSession()

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


def model_batch_norm(x, y_, reuse, is_train):
    net = InputLayer(x, name='input')
    net = Conv2d(net, 64, (5, 5), (1, 1),  name='cnn1')
    net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch1')
    net = MaxPool2d(net, (3, 3), (2, 2), name='pool1')

    net = Conv2d(net, 64, (5, 5), (1, 1),  name='cnn2')
    net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch2')
    net = MaxPool2d(net, (3, 3), (2, 2), name='pool2')

    net = FlattenLayer(net, name='flatten')                             
    net = DenseLayer(net, n_units=384, act=tf.nn.relu,  name='d1relu')           
    net = DenseLayer(net, n_units=192, act = tf.nn.relu,  name='d2relu')           
    net = DenseLayer(net, n_units=10, act = tf.identity,  name='output')  
    y = net.outputs

    ce = tl.cost.cross_entropy(y, y_, name='cost')
    L2 = 0
    for p in tl.layers.get_variables_with_name('relu/W', True, True):
        L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
    cost = ce + L2

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return net, cost, acc
 

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
network, cost, acc = model_batch_norm(x, y_, False, is_train=True)

## train
n_epoch = 50000
learning_rate = 0.0001
print_freq = 1
batch_size = 128

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)

network.print_params(False)
network.print_layers()


print(X_train.shape,y_train.shape)
print("fit......")
# 训练
# tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
#                  acc=acc, batch_size=batch_size, n_epoch=n_epoch, print_freq=print_freq, X_val=X_test, y_val=y_test, eval_train=True)

# for epoch in range(n_epoch):
#     start_time = time.time()
#     for X_train_a, y_train_a in tl.iterate.minibatches(
#                                 X_train, y_train, batch_size, shuffle=True):
#         X_train_a = tl.prepro.threading_data(X_train_a, fn=distort_fn, is_train=True)  # data augmentation for training
#         sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

#     if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#         print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
#         train_loss, train_acc, n_batch = 0, 0, 0
#         for X_train_a, y_train_a in tl.iterate.minibatches(
#                                 X_train, y_train, batch_size, shuffle=True):
#             X_train_a = tl.prepro.threading_data(X_train_a, fn=distort_fn, is_train=False)  # central crop
#             err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
#             train_loss += err; train_acc += ac; n_batch += 1
#         print("   train loss: %f" % (train_loss/ n_batch))
#         print("   train acc: %f" % (train_acc/ n_batch))
#         test_loss, test_acc, n_batch = 0, 0, 0
#         for X_test_a, y_test_a in tl.iterate.minibatches(
#                                     X_test, y_test, batch_size, shuffle=False):
#             X_test_a = tl.prepro.threading_data(X_test_a, fn=distort_fn, is_train=False)   # central crop
#             err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
#             test_loss += err; test_acc += ac; n_batch += 1
#         print("   test loss: %f" % (test_loss/ n_batch))
# print("   test acc: %f" % (test_acc/ n_batch))