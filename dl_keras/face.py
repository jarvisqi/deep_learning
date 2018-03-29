import base64
import struct
import os
import glob
import yaml
import pandas as pd
import numpy as np
from keras import regularizers
from keras.models import Sequential, load_model, model_from_yaml
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils, plot_model
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

np.random.seed(1024)

# 参数设置
image_size = (100, 100)
image_row = 100
nb_classes = 423
batch_size = 256
nb_epoch = 32

labelName = []



def preprocess():
    '''
    数据预处理
    '''
    count = 0
    total = 0
    path = './data/faces/'
    for f in os.listdir(path):
        name = f.split(' ')[0]
        os.makedirs(path + name)
        oldPath = path + f
        newPath = path + name + '/' + f
        shutil.copyfile(oldPath, newPath)

def load_data():
    path = './data/faces/'
    count = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]

    images = []
    labels = []
    for idx, folder in enumerate(count):
        for img_path in glob.glob(folder + '/*.jpg'):
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(idx)

    images = np.array(images)
    labels = np.array(labels)

    labels = np.reshape(labels, (len(labels), 1))
    labels = np_utils.to_categorical(labels, nb_classes)

    # 打乱顺序
    num = images.shape[0]
    arr = np.arange(num)
    np.random.shuffle(arr)
    images = images[arr]
    labels = labels[arr]

    print(images.shape)
    print(labels.shape)

    return images, labels


def build_model():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid',
                     input_shape=(image_row, image_row, 3), activation='relu'))
    #  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.001)))
    model.add(Conv2D(42, kernel_size=(3, 3), activation='relu'))
    #  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.001)))
    # 池化层往往在卷积层后面，通过池化来降低卷积层输出的特征向量，同时改善结果（不易出现过拟合）。
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #                 # kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    print('编译模型...')
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def train_model(model: Sequential):
    data, lable = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        data, lable, test_size=0.1)
    x_train /= 255
    x_test /= 255
    print(len(x_train), len(y_train), len(x_test), len(y_test))
    print("训练...")

    # # early_stopping = EarlyStopping(monitor='val_loss', patience=2) # , callbacks=[early_stopping]
    # # model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
    # #           verbose=1, validation_data=(x_test, y_test))

    train_datagen = image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2,
                                             horizontal_flip=True)
    validation_datagen = image.ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        './data/faces/', target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=True)
    validation_generator = validation_datagen.flow_from_directory(
        './data/faces/', target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=True)

    model.fit_generator(train_generator, steps_per_epoch=5985, epochs=nb_epoch, validation_data=validation_generator,
                        validation_steps=599)

    print("评估...")
    score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', accuracy)

    yaml_string = model.to_yaml()
    with open('./models/face.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./models/face_weight.h5')


def pred_data():

    with open('./models/face.yaml') as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('./models/face_weight.h5')

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    plot_model(model,to_file='./models/face_model.png')


if __name__ == '__main__':

    model = build_model()
    train_model(model)

    # load_data()

    # pred_data()
