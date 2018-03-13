import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn import metrics
# 使用tf.contrib.layers中定义好的卷积神经网络结构可以更方便的实现卷积层。
layers = tf.contrib.layers
learn = tf.contrib.learn
# 自定义模型结构。这个函数有三个参数，第一个给出了输入的特征向量，第二个给出了
# 该输入对应的正确输出，最后一个给出了当前数据是训练还是测试。该函数的返回也有
# 三个指，第一个为定义的神经网络结构得到的最终输出节点取值，第二个为损失函数，第
# 三个为训练神经网络的操作。


def conv_model(input, target, mode):
     # 将正确答案转化成需要的格式。
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    # 定义神经网络结构，首先需要将输入转化为一个三维举证，其中第一维表示一个batch中的
    # 样例数量。
    network = tf.reshape(input, [-1, 28, 28, 1])
    # 通过tf.contrib.layers来定义过滤器大小为5*5的卷积层。
    network = layers.convolution2d(network, 32, kernel_size=[
                                   5, 5], activation_fn=tf.nn.relu)
    # 实现过滤器大小为2*2，长和宽上的步长都为2的最大池化层。
    network = tf.nn.max_pool(network, ksize=[1, 2, 2, 1], strides=[
                             1, 2, 2, 1], padding='SAME')
    # 类似的定义其他的网络层结构。
    network = layers.convolution2d(network, 64, kernel_size=[
                                   5, 5], activation_fn=tf.nn.relu)
    network = tf.nn.max_pool(network, ksize=[1, 2, 2, 1], strides=[
                             1, 2, 2, 1], padding='SAME')
    # 将卷积层得到的矩阵拉直成一个向量，方便后面全连接层的处理。
    network = tf.reshape(network, [-1, 7 * 7 * 64])
    # 加入dropout。注意dropout只在训练时使用。
    network = layers.dropout(
        layers.fully_connected(network, 500, activation_fn=tf.nn.relu),
        keep_prob=0.5,
        is_training=(mode == tf.contrib.learn.ModeKeys.TRAIN))
    # 定义最后的全连接层。
    logits = layers.fully_connected(network, 10, activation_fn=None)
    # 定义损失函数。
    loss = tf.losses.softmax_cross_entropy(target, logits)
    # 定义优化函数和训练步骤。
    train_op = layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='SGD',
        learning_rate=0.01)
    return tf.argmax(logits, 1), loss, train_op


def main():

    # 加载数据。
    mnist = learn.datasets.load_dataset('mnist')
    # 定义神经网络结构，并在训练数据上训练神经网络。
    classifier = learn.Estimator(model_fn=conv_model)
    classifier.fit(mnist.train.images, mnist.train.labels,
                   batch_size=100, steps=20000)
    # 在测试数据上计算模型准确率。
    score = metrics.accuracy_score(mnist.test.labels, list(
        classifier.predict(mnist.test.images)))
    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
    main()
