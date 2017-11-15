# -*- coding:utf-8 -*-
# 多输出车牌识别

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from genplate import GenPlate
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.utils import vis_utils,plot_model,np_utils
from keras.preprocessing import image
import matplotlib
from matplotlib import pyplot as plt
from IPython.display import SVG
import cv2

np.random.seed(5)
image_size = (272, 72)
nbatch_size = 32
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
M_strIdx = dict(zip(chars, range(len(chars))))


def gen_plate():
    """
    车牌生成器
    """
    G = GenPlate("./font/platech.ttf", './font/platechar.ttf', "./images/NoPlates")
    while True:
        l_plateStr, l_plateImg = G.genBatch(1, 2, range(31, 65), "./images/plate", image_size)
        X = np.array(l_plateImg, dtype=np.uint8)
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_plateStr)), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1], nbatch_size, len(chars)])
        for batch in range(nbatch_size):
            for idx, row_i in enumerate(ytmp[batch]):
                y[idx, batch, row_i] = 1

        yield X, [yy for yy in y]


def main():
    input_tensor = Input((72, 272, 3))
    x = input_tensor
    print("build model")
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    n_class = len(chars)
    x = [Dense(n_class, activation='softmax', name='c{0}'.format(i + 1))(x) for i in range(7)]
    model = Model(inputs=input_tensor, outputs=x)
    
    print("compile model")
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # display 
    plot_model(model,to_file='./models/licenseplate_model.png')

    # training
    print("training model")
    best_model = ModelCheckpoint("./models/licenseplate.h5", monitor='val_loss', verbose=0, save_best_only=True)
    model.fit_generator(gen_plate(), steps_per_epoch=2000, epochs=8, validation_data=gen_plate(),
                     validation_steps=1280, callbacks=[best_model])


def  predict_plate():
    # G = GenPlate("./font/platech.ttf", './font/platechar.ttf', "./images/NoPlates")
    # l_plateStr, l_plateImg = G.genBatch(10, 2, range(31, 65), "./images/plate_test", image_size)
    # x_test = np.array(l_plateImg, dtype=np.uint8)
    # print(x_test.shape)

    img = cv2.imread("../predict_img/TF521.jpg")  
    img = cv2.resize(img, image_size)
    x_test = np.array([img], dtype=np.uint8)
    print(x_test.shape)

    model = load_model('./models/licenseplate.h5')
    l_titles = list(map(lambda x: "".join([chars[xx] for xx in x]), np.argmax(np.array(model.predict(x_test)), 2).T))
    print(l_titles)



if __name__ == '__main__':
    
    # main()

    predict_plate()


