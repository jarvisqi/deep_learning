import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input, decode_predictions

np.random.seed(7)
img_h, img_w = 150, 150
image_size = (150, 150)
nbatch_size = 256
nepochs = 48
nb_classes = 2


def load_data():
    path = './data/train/'
    files = os.listdir(path)
    images = []
    labels = []
    for f in files:
        img_path = path + f
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        images.append(img_array)

        if 'cat' in f:
            labels.append(0)
        else:
            labels.append(1)

    data = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, 2)
    return data, labels


def main():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(img_h, img_h, 3), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.summary()

    print("compile.......")
    sgd = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])

    print("load_data......")
    # images, lables = load_data()
    # images, lables = load_data()
    images = np.load("./data/cd_data.npy")
    labels = np.load("./data/cd_label.npy")
    labels = np_utils.to_categorical(labels, 2)
    images /= 255

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    print(x_train.shape,y_train.shape)

    print("train.......")
    tbCallbacks = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs, verbose=1, validation_data=(x_test, y_test), callbacks=[tbCallbacks])

    print("evaluate......")
    scroe, accuracy = model.evaluate(x_test, y_test, batch_size=nbatch_size)
    print('scroe:', scroe, 'accuracy:', accuracy)

    yaml_string = model.to_yaml()
    with open('./models/cat_dog.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./models/cat_dog.h5')


def pred_data():

    with open('./models/cat_dog.yaml') as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('./models/cat_dog.h5')

    sgd = Adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

    images = []
    path='./data/test/'
    for f in os.listdir(path):
        img = image.load_img(path + f, target_size=image_size)
        img_array = image.img_to_array(img)

        x = np.expand_dims(img_array, axis=0)
        x = preprocess_input(x)
        result = model.predict_classes(x,verbose=0)

        print(f,result[0])
    

if __name__ == '__main__':

    # main()

    # pred_data()

    # load_data()

    pred_data()
