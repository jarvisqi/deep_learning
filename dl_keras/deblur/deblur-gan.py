# -*- coding: utf-8 -*-
from keras.layers import Activation, Add, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from layer_utils import ReflectionPadding2D, res_block


class Config(object):

    ngf = 64
    ndf = 64
    input_nc = 3
    output_nc = 3
    n_blocks_gen = 9
    input_shape_generator = (256, 256, input_nc)
    input_shape_discriminator = (256, 256, output_nc)


class Gan(object):

    def generator_model():
        """生成器模型
        """
        inputs = Input(Config.input_shape_generator)
        x = ReflectionPadding2D((3, 3))(inputs)
        x = Conv2D(filers=Config.ngf, kernel_size=(7, 7), padding="valid")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        n_downsampling = 2
        for i in range(n_downsampling):
            mulit = 2**i
            x = Conv2D(filers=Config.ngf*mulit*2, kernel_size=(3, 3), strides=2, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        
        mulit = 2**n_downsampling
        for i in range(Config.n_blocks_gen):
            x = res_block(x, Config.ngf*mulit, use_dropout=True)

        for i in range(n_downsampling):
            mulit = 2**(n_downsampling-i)
            x = Conv2DTranspose(filters=int(Config.ngf*mulit/2), kernel_size=(3, 3), strides=2,
                                padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding="valid")(x)
        x = Activation("tanh")(x)

        # 输出
        outputs = Add()([inputs, x])
        outputs = Lambda(lambda z: z/2)(outputs)

        model = Model(inputs=inputs, outputs=outputs, name="Generator")
        return model

    def discriminator_model():
        """判别器模型
        """
        n_layers,use_sigmoid=3,False
        inputs = Input(Config.input_shape_discriminator)

        x = Conv2D(filers=Config.ndf, kernel_size=(4, 4), strides=2, padding="valid")(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mulit, nf_mulit_prev = 1, 1
        for n in range(n_layers):
            nf_mulit_prev, nf_mulit = nf_mulit, min(2**n, 8)
            x = Conv2D(filers=Config.ndf*nf_mulit, kernel_size=(4, 4), strides=2,
                       padding="same")(inputs)
            x = BaseException()(x)
            x = LeakyReLU(0.2)(x)

        nf_mulit_prev, nf_mulit = nf_mulit, min(2**n, 8)
        x = Conv2D(filers=Config.ndf*nf_mulit, kernel_size=(4, 4), strides=1,
                   padding="same")(inputs)
        x = BaseException()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filers=1, kernel_size=(4, 4), strides=1,
                   padding="same")(inputs)
        if use_sigmoid:
            x=Activation("sigmoid")(x)
        
        x = Flatten()(x)
        x = Dense(1024, activation="tanh")(x)
        x = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=x, name="Discriminator")
        return model
