import os

from keras import layers
from keras.layers import Input, merge
from keras.layers.convolutional import (AveragePooling2D, Conv2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def identity_block(x, nb_filter, kernel_size=3):
    k1, k2, k3 = nb_filter

    shortcut = x 
    out = Conv2D(k1, kernel_size=(1,1), strides=(1,1),padding="valid",activation="relu")(x)
    out = BatchNormalization(axis=3)(out)

    out = Conv2D(k2, kernel_size=(3,3), strides=(1,1), padding='same',activation="relu")(out)
    out = BatchNormalization(axis=3)(out)

    out = Conv2D(k3, kernel_size=(1,1), strides=(1,1),padding="valid")(out)
    out = BatchNormalization(axis=3)(out)

    # out = merge([out, shortcut], mode='sum')
    out= layers.add([out,shortcut])  
    out = Activation('relu')(out)
    return out


def conv_block(x, nb_filter, kernel_size=3):
    k1, k2, k3 = nb_filter
    shortcut = x 
    
    out = Conv2D(k1, kernel_size=(1,1), strides=(2,2), padding="valid",activation="relu")(x)
    out = BatchNormalization(axis=3)(out)

    out = out = Conv2D(k2, kernel_size=(kernel_size,kernel_size), strides=(1,1), padding="same",activation="relu")(out)
    out = BatchNormalization()(out)

    out = Conv2D(k3, kernel_size=(1,1), strides=(1,1), padding="valid")(out)
    out = BatchNormalization(axis=3)(out)

    shortcut = Conv2D(k3, kernel_size=(1,1), strides=(2,2), padding="valid")(shortcut)
    shortcut = BatchNormalization(axis=3)(shortcut)

    # out = merge([out, shortcut], mode='sum')
    out = layers.add([out, shortcut])
    out = Activation('relu')(out)
    return out


def buildNet():
    inp = Input(shape=(224, 224, 3))

    out = ZeroPadding2D((3, 3))(inp)
    out = Conv2D(64, kernel_size=(7, 7), strides=(2, 2),activation="relu")(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(out)

    out = conv_block(out, [64, 64, 256])
    out = identity_block(out, [64, 64, 256])
    out = identity_block(out, [64, 64, 256])

    out = conv_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])

    out = conv_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])
    out = identity_block(out, [256, 256, 1024])

    out = conv_block(out, [512, 512, 2048])
    out = identity_block(out, [512, 512, 2048])
    out = identity_block(out, [512, 512, 2048])

    out = AveragePooling2D((4, 4))(out)
    out = Flatten()(out)  # 展平
    out = Dense(1000, activation='softmax')(out)

    model = Model(inputs=inp, outputs=out)
    return model





if __name__ == '__main__':

    # resNet18 = ResNet(block_num=[2,2,2,2])
    # resNet34 = ResNet(block_num=[3,4,6,3])
    # resNet50 = ResNet(block_num=[3,4,6,3])
    # resNet101 = ResNet(block_num=[3,4,23,3])
    # resNet152 = ResNet(block_num=[3,8,36,3])

    net = buildNet()
    net.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(net, to_file='./models/resnet.png')
    net.summary()
