import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from skimage import io, transform
from keras.preprocessing import image as kimg
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


np.random.seed(1024)
# np.random.seed(19260816)
img_h, img_w, img_c = 120, 120, 3
n_classes = 2
kernel_size=(5,5)

# 定义  placeholder 占位符
x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, img_c], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')


def load_data():
    """
    加载数据
    """
    path = './data/train/'
    files = os.listdir(path)
    images = []
    labels = []
    for f in files:
        img = kimg.load_img(path + f, target_size=(img_h, img_w))
        img_array = kimg.img_to_array(img)
        images.append(img_array)
        if 'cat' in f:
            labels.append(0)
        else:
            labels.append(1)

    data = np.array(images)
    labels = np.array(labels)

    if os.path.exists("./data/cd_data.npy"):
        os.remove("./data/cd_data.npy")
        os.remove("./data/cd_label.npy")
    np.save("./data/cd_data.npy", data)
    np.save("./data/cd_label.npy", labels)

    return data, labels


def build_net(x):
    """
    构建网络
    """
    print("build network")
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, n_filter=32, filter_size=kernel_size, act=tf.nn.relu, name='conv2d1')
    network = tl.layers.MaxPool2d(network, name='maxpool1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')

    network = tl.layers.Conv2d(network, n_filter=64, filter_size=kernel_size, act=tf.nn.relu, name='conv2d2')
    network = tl.layers.MaxPool2d(network, name='maxpool2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')

    network = tl.layers.Conv2d(network, n_filter=128, filter_size=kernel_size, act=tf.nn.relu, name='conv2d3')
    network = tl.layers.MaxPool2d(network, name='maxpool3')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout3')

    network = tl.layers.Conv2d(network, n_filter=256, filter_size=kernel_size, act=tf.nn.relu, name='conv2d4')
    network = tl.layers.MaxPool2d(network, name='maxpool4')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout4')

    network = tl.layers.FlattenLayer(network, name='flatten_layer')
    network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='dense_layer1')
    network = tl.layers.DenseLayer(network, n_units=256, act=tf.nn.relu, name='dense_layer2')
    network = tl.layers.DenseLayer(network, n_units=128, act=tf.nn.relu, name='dense_layer3')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout5')

    network = tl.layers.DenseLayer(network, n_units=n_classes, act=tf.identity, name='output')

    return network


def cost_fn(network):
    # 定义损失函数
    y = network.outputs
    # 交叉熵损失函数
    cost = tl.cost.cross_entropy(y, y_, name='cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return cost, acc


def train():
    """
    训练模型
    """
    sess = tf.InteractiveSession()

    print("load data......")
    # images, lables = load_data()
    images = np.load("./data/cd_data.npy")
    labels = np.load("./data/cd_label.npy")
    images /= 255
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    network = build_net(x)
    cost, acc = cost_fn(network)

    # 定义优化器
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(cost, var_list=train_params)

    tf.summary.scalar('acc', acc)
    # tf.summary.scalar('cost', cost)
    # 初始化 Session 中所有参数
    tl.layers.initialize_global_variables(sess)
    # 打印模型和参数信息
    network.print_layers()
    network.print_params()

    print("fit......")
    # 训练
    tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
                 acc=acc, batch_size=256, n_epoch=64, print_freq=2, X_val=X_test, y_val=y_test,
                 tensorboard=True)
    print("test......")
    # 评估模型
    tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=256)
    tl.files.save_ckpt(sess=sess,mode_name='tl_catdog_model.ckpt',save_dir='./models/checkpoint')

    sess.close()


def test_predict():

    sess = tf.InteractiveSession()
    network = build_net(x)

    # 加载模型
    tl.files.load_and_assign_npz(sess, name="./models/model.npz", network=network)
    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    img = read_one_image("./predict_img/cat_dog/c_1.jpg")
    x_test = np.array([img.reshape(-1)])
    tx = tf.convert_to_tensor(img)
    result = tl.utils.predict(sess, network, x_test, x, y_op)
    print(result)

    sess.close()


if __name__ == '__main__':

    print(x.shape, y_.shape)

    # load_data()

    train()
