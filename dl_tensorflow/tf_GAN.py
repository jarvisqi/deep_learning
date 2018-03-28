# -*- coding: utf-8 -*-
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_image(c_set):
    """
    # 生成字符对应的验证码
    """
    image = ImageCaptcha(width=120, height=60)
    captcha_text = random_captcha_text(char_set=c_set)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image


class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)


# 生成长度为4的验证码
captcha_len = 4
image_size = [64, 128]  # 最好是2的次方
batch_size = 64
# noise
z_dim = 20
t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name="noise")
t_dim = 80
# 验证码字符串
t_text = tf.placeholder(
    tf.float32, [batch_size, captcha_len], name="captcha_text")

# t_text字符串对应的验证码
real_image = tf.placeholder(
    tf.float32, [batch_size, image_size[0], image_size[1], 3], name='real_image')
# t_text字符串对应的错误验证码
wrong_image = tf.placeholder(
    tf.float32, [batch_size, image_size[0], image_size[1], 3], name='wrong_image')


def generator(t_z, t_text, is_training=True):
    """
    生成器
    """
    with tf.variable_scope("g_captcha_text_embdding"):
        W = tf.get_variable("g_weight", [
            captcha_len, t_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            "g_bias", [t_dim], initializer=tf.constant_initializer(0.0))
        captcha_text_embedding = tf.matmul(t_text, W) + b  # (batch_size, 80)
        # leak relu
        captcha_text_embedding = tf.maximum(
            captcha_text_embedding, 0.2 * captcha_text_embedding)
        # (batch_size, 100)
        z_concat = tf.concat([t_z, captcha_text_embedding], 1)

    w_2, w_4, w_8, w_16 = int(
        image_size[0] / 2), int(image_size[0] / 4), int(image_size[0] / 8), int(image_size[0] / 16)
    h_2, h_4, h_8, h_16 = int(
        image_size[1] / 2), int(image_size[1] / 4), int(image_size[1] / 8), int(image_size[1] / 16)

    with tf.variable_scope("g_projection_layer"):
        W = tf.get_variable("g_weight", [
            z_dim + t_dim, 64 * 8 * w_16 * h_16], tf.float32, tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            "g_bias", [64 * 8 * w_16 * h_16], initializer=tf.constant_initializer(0.0))
        out = tf.nn.relu(tf.matmul(z_concat, W) + b)
        out = tf.reshape(out, [-1, w_16, h_16, 64 * 8])
        # (64, 4, 8, 512)

    with tf.variable_scope("g_deconv2d"):
        W1 = tf.get_variable('g_deconv2d_W_1', [
            5, 5, 64 * 4, 64 * 8], initializer=tf.random_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('g_deconv2d_b_1', [
            64 * 4], initializer=tf.constant_initializer(0.0))
        deconv_1 = tf.nn.conv2d_transpose(
            out, W1, output_shape=[batch_size, w_8, h_8, 64 * 4], strides=[1, 2, 2, 1])
        deconv_1 = tf.nn.bias_add(deconv_1, b1)
        deconv_1_batch_norm_func = batch_norm(name='g_deconv2d_1_bn')
        deconv_1 = tf.nn.relu(deconv_1_batch_norm_func(
            deconv_1, train=is_training))
        # (?, 8, 16, 256)

        W2 = tf.get_variable('g_deconv2d_W_2', [
            5, 5, 64 * 2, 64 * 4], initializer=tf.random_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('g_deconv2d_b_2', [
            64 * 2], initializer=tf.constant_initializer(0.0))
        deconv_2 = tf.nn.conv2d_transpose(deconv_1, W2, output_shape=[
            batch_size, w_4, h_4, 64 * 2], strides=[1, 2, 2, 1])
        deconv_2 = tf.nn.bias_add(deconv_2, b2)
        deconv_2_batch_norm_func = batch_norm(name='g_deconv2d_2_bn')
        deconv_2 = tf.nn.relu(deconv_2_batch_norm_func(
            deconv_2, train=is_training))
        # (?, 16, 32, 128)

        W3 = tf.get_variable('g_deconv2d_W_3', [
            5, 5, 64, 64 * 2], initializer=tf.random_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('g_deconv2d_b_3', [
            64], initializer=tf.constant_initializer(0.0))
        deconv_3 = tf.nn.conv2d_transpose(deconv_2, W3, output_shape=[
            batch_size, w_2, h_2, 64], strides=[1, 2, 2, 1])
        deconv_3 = tf.nn.bias_add(deconv_3, b3)
        deconv_3_batch_norm_func = batch_norm(name='g_deconv2d_3_bn')
        deconv_3 = tf.nn.relu(deconv_3_batch_norm_func(
            deconv_3, train=is_training))
        # (?, 32, 64, 64)

        W4 = tf.get_variable('g_deconv2d_W_4', [
            5, 5, 3, 64], initializer=tf.random_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('g_deconv2d_b_4', [
            3], initializer=tf.constant_initializer(0.0))
        deconv_4 = tf.nn.conv2d_transpose(deconv_3, W4, output_shape=[
            batch_size, image_size[0], image_size[1], 3], strides=[1, 2, 2, 1])
        deconv_4 = tf.nn.bias_add(deconv_4, b4)
        # (?, 64, 128, 3)
        return tf.tanh(deconv_4) / 2. + 0.5  # tanh范围(-1 1), 转为(0 1)


def discriminator(image, t_text, reuse=False):
    """
    判别器
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("d_conv2d"):
        W1 = tf.get_variable('d_conv2d_W_1', [5, 5, image.get_shape(
        )[-1], 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable(
            'd_conv2d_b_1', [64], initializer=tf.constant_initializer(0.0))
        conv_1 = tf.nn.conv2d(image, W1, strides=[1, 2, 2, 1], padding='SAME')
        conv_1 = tf.nn.bias_add(conv_1, b1)
        conv_1 = tf.maximum(conv_1, 0.2 * conv_1)
        # (64, 32, 64, 64)

        W2 = tf.get_variable('d_conv2d_W_2', [5, 5, conv_1.get_shape(
        )[-1], 64 * 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable(
            'd_conv2d_b_2', [64 * 2], initializer=tf.constant_initializer(0.0))
        conv_2 = tf.nn.conv2d(conv_1, W2, strides=[1, 2, 2, 1], padding='SAME')
        conv_2 = tf.nn.bias_add(conv_2, b2)
        conv_2_batch_norm_func = batch_norm(name='d_conv2d_2_bn')
        conv_2 = conv_2_batch_norm_func(conv_2, train=True)
        conv_2 = tf.maximum(conv_2, 0.2 * conv_2)
        # (64, 16, 32, 128)

        W3 = tf.get_variable('d_conv2d_W_3', [5, 5, conv_2.get_shape(
        )[-1], 64 * 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable(
            'd_conv2d_b_3', [64 * 4], initializer=tf.constant_initializer(0.0))
        conv_3 = tf.nn.conv2d(conv_2, W3, strides=[1, 2, 2, 1], padding='SAME')
        conv_3 = tf.nn.bias_add(conv_3, b3)
        conv_3_batch_norm_func = batch_norm(name='d_conv2d_3_bn')
        conv_3 = conv_3_batch_norm_func(conv_3, train=True)
        conv_3 = tf.maximum(conv_3, 0.2 * conv_3)
        # (64, 8, 16, 256)

        W4 = tf.get_variable('d_conv2d_W_4', [5, 5, conv_3.get_shape(
        )[-1], 64 * 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable(
            'd_conv2d_b_4', [64 * 8], initializer=tf.constant_initializer(0.0))
        conv_4 = tf.nn.conv2d(conv_3, W4, strides=[1, 2, 2, 1], padding='SAME')
        conv_4 = tf.nn.bias_add(conv_4, b4)
        conv_4_batch_norm_func = batch_norm(name='d_conv2d_4_bn')
        conv_4 = conv_4_batch_norm_func(conv_4, train=True)
        conv_4 = tf.maximum(conv_4, 0.2 * conv_4)
        # (64, 4, 8, 512)

        with tf.variable_scope("d_captcha_text_embedding"):
            W = tf.get_variable("d_weight", [
                captcha_len, t_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            b = tf.get_variable(
                "d_bias", [t_dim], initializer=tf.constant_initializer(0.0))
            captcha_text_embedding = tf.matmul(
                t_text, W) + b  # (batch_size, 80)
            # leak relu
            captcha_text_embedding = tf.maximum(
                captcha_text_embedding, 0.2 * captcha_text_embedding)
            captcha_text_embedding = tf.expand_dims(captcha_text_embedding, 1)
            captcha_text_embedding = tf.expand_dims(captcha_text_embedding, 2)
            tiled_embeddings = tf.tile(captcha_text_embedding, [
                1, 4, 8, 1], name='d_tiled_embeddings')

        conv_4_concat = tf.concat(
            [conv_4, tiled_embeddings], 3, name='d_conv_4_concat')
        # (64, 4, 8, 592)

        W5 = tf.get_variable('d_conv2d_W_5', [1, 1, conv_4_concat.get_shape(
        )[-1], 64 * 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b5 = tf.get_variable(
            'd_conv2d_b_5', [64 * 8], initializer=tf.constant_initializer(0.0))
        conv_5 = tf.nn.conv2d(conv_4_concat, W5, strides=[
            1, 1, 1, 1], padding='SAME')
        conv_5 = tf.nn.bias_add(conv_5, b5)
        conv_5_batch_norm_func = batch_norm(name='d_conv2d_5_bn')
        conv_5 = conv_5_batch_norm_func(conv_5, train=True)
        conv_5 = tf.maximum(conv_5, 0.2 * conv_5)
        # (64, 4, 8, 512)

        with tf.variable_scope("d_fully_connect"):
            flat = tf.reshape(conv_5, [batch_size, 4 * 8 * 512])
            W = tf.get_variable("d_W", [flat.get_shape().as_list()[
                                            1], 1], initializer=tf.random_normal_initializer(stddev=0.02))
            b = tf.get_variable(
                "d_b", [1], initializer=tf.constant_initializer(0.0))
            fc = tf.matmul(flat, W) + b

            return tf.nn.sigmoid(fc), fc




def get_next_batch(batch_size=64):
    """
    # 生成一个训练batch
    """
    # 有时生成图像大小不是(60, 160, 3)
    batch_texts = []
    batch_images = []
    batch_wrong_images = []

    def wrap_gen_captcha_text_and_image(c_set):
        while True:
            text, image = gen_captcha_text_image(c_set)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image(c_set=['1', '2', '3', '4'])
        ord_n = []
        for c in text:
            ord_n.append(ord(c) / 122.0)

        image = Image.fromarray(image)
        image = image.resize([128, 64])
        # image.save(str(i)+'.jpg')
        image = np.array(image) / 255.0

        _, wrong_image = wrap_gen_captcha_text_and_image(c_set=['a', 'b', 'c', 'd'])
        wrong_image = Image.fromarray(wrong_image)
        wrong_image = wrong_image.resize([128, 64])
        wrong_image = np.array(wrong_image) / 255.0

        batch_texts.append(ord_n)
        batch_images.append(image)
        batch_wrong_images.append(wrong_image)
    return np.array(batch_texts), np.array(batch_images), np.array(batch_wrong_images)


def train():
    fake_image = generator(t_z, t_text, is_training=True)

    fake_image_disc, fake_image_logits_disc = discriminator(fake_image, t_text)
    wrong_image_disc, wrong_image_logits_disc = discriminator(wrong_image, t_text, reuse=True)
    real_image_disc, real_image_logits_disc = discriminator(real_image, t_text, reuse=True)
    # loss
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_image_logits_disc, logits=tf.ones_like(fake_image_disc)))

    d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_image_logits_disc, logits=tf.ones_like(real_image_disc)))
    d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=wrong_image_logits_disc, logits=tf.zeros_like(wrong_image_disc)))
    d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_image_logits_disc, logits=tf.zeros_like(fake_image_disc)))
    d_loss = d_loss1 + d_loss2 + d_loss3

    train_vars = tf.trainable_variables()
    d_vars = [var for var in train_vars if 'd_' in var.name]
    g_vars = [var for var in train_vars if 'g_' in var.name]

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars)
    
    print("run")
    with tf.Session() as sess:
        loop = 0
        while True:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            print(loop)
            # saver.restore(sess, '')
            batch_text, batch_image, batch_wrong_images = get_next_batch(64)
            z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

            _, _d_loss = sess.run([d_optim, d_loss], feed_dict={real_image: batch_image,
                                                                wrong_image: batch_wrong_images,
                                                                t_z: z_noise,
                                                                t_text: batch_text})
            # 更新generator两次，防止d_loss->0
            _, _ = sess.run([g_optim, g_loss], feed_dict={real_image: batch_image,
                                                          wrong_image: batch_wrong_images,
                                                          t_z: z_noise,
                                                          t_text: batch_text})
            _, _g_loss = sess.run([g_optim, g_loss], feed_dict={real_image: batch_image,
                                                                wrong_image: batch_wrong_images,
                                                                t_z: z_noise,
                                                                t_text: batch_text})

            print(loop, _d_loss, _g_loss)

            if loop % 50 == 0:
                save_path = saver.save(sess, "./models/tf_GAN/captcha.model")
                texts, _, _ = get_next_batch(64)
                z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
                images = sess.run(fake_image, feed_dict={t_z: z_noise, t_text: texts})

                os.mkdir(str(loop))
                i = 0
                for img in images:
                    plt.imsave("./images/tf_GAN/"+str(loop) + '/' + str(i) + '.jpg', img)
                    i += 1

            loop += 1


if __name__ == '__main__':
    train()
