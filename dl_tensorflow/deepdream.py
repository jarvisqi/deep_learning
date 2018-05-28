import os
from functools import partial
from io import BytesIO

import numpy as np
import PIL.Image
import scipy.misc
import tensorflow as tf

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
model_fn = "./models/tensorflow_inception_graph.pb"
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(tf.float32, name="input")
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {"input": t_preprocessed})


def load_inception():
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    model_fn = "./models/tensorflow_inception_graph.pb"
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 定义t_input为我们输入的图像
    t_input = tf.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    # 输入图像需要经过处理才能送入网络中
    # expand_dims是加一维，从[height, width, channel]变成[1, height, width, channel]
    # t_input - imagenet_mean是减去一个均值
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})
    # 找到所有卷积层
    layers = [op.name for op in graph.get_operations() if op.type ==
            "Conv2D" and "import/" in op.name]

    # 输出卷积层层数
    print('Number of layers', len(layers))
    # 特别地，输出mixed4d_3x3_bottleneck_pre_relu的形状
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    print('shape of %s: %s' %(name, str(graph.get_tensor_by_name('import/' + name + ':0').get_shape())))


def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


def visstd(a, s=0.1):
    return (a-a.mean())/max(a.std(), 1e-4)*s+0.5

def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img / 255 * (max - min) + min
    return img


def resize(img, hw):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, hw))
    img = img / 255 * (max - min) + min
    return img


def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)  # 先在行上做整体移动，再在列上做整体移动
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)

# 将拉普拉斯金字塔还原到原始图像
def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
    return img

# 对img做标准化。
def normalize_std(img, eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, eps)

# 拉普拉斯金字塔标准化
def lap_normalize(img, scale_n=4):
    img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, scale_n)
    # 每一层都做一次normalize_std
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0, :, :, :]


# 这个函数将图像分为低频和高频成分
def lap_split(img):
    with tf.name_scope('split'):
        # 做过一次卷积相当于一次“平滑”，因此lo为低频成分
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
        # 低频成分放缩到原始图像一样大小得到lo2，再用原始图像img减去lo2，就得到高频成分hi
        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
        hi = img - lo2
    return lo, hi


# 这个函数将图像img分成n层拉普拉斯金字塔
def lap_split_n(img, n):
    levels = []
    for i in range(n):
        # 调用lap_split将图像分为低频和高频部分
        # 高频部分保存到levels中
        # 低频部分再继续分解
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]


def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


def render_deepdream(img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    t_obj = graph.get_tensor_by_name("import/%s:0" % name)
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    lap_n=4
    # 将lap_normalize转换为正常函数
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0
    # 同样将图像进行金字塔分解
    # 此时提取高频、低频的方法比较简单。直接缩放就可以
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # 先生成低频的图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
            # 唯一的区别在于我们使用lap_norm_func来标准化g！
            # g = lap_norm_func(g)
            # img += g * step
            print('.', end=' ')
    
    img = img.clip(0, 255)
    savearray(img, './predict_img/deepdream.jpg')


if __name__ == '__main__':
    img0 = PIL.Image.open('./images/test.jpg')
    img0 = np.float32(img0)

    render_deepdream(img0)
