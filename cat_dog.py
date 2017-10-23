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
image_size = (128, 128)
nbatch_size = 128
nepochs = 64
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

    # labels = np.reshape(labels, (len(labels), 1))
    labels = np_utils.to_categorical(labels, nb_classes)

    return data, labels


def main():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),input_shape=(128, 128, 3), activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(5,5), activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()

    print("compile.......")
    sgd = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    print("load_data......")
    images, lables = load_data()
    images /=255
    x_train, x_test, y_train, y_test = train_test_split(images, lables, test_size=0.15)
    print("x_train:",x_train.shape)

    print("train.......")
    tbCallbacks = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs, verbose=1, validation_data=(x_test, y_test), callbacks=[tbCallbacks])

    print("evaluate......")
    scroe, accuracy = model.evaluate(x_test, y_test, batch_size=nbatch_size)
    print('scroe:', scroe, 'accuracy:', accuracy)

    yaml_string = model.to_yaml()
    with open('./data/text/cat_dog.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./data/text/cat_dog.h5')

def pred_data():
    
    with open('./models/cat_dog.yaml') as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('./models/cat_dog.h5')

    sgd = Adam(lr=0.0001)
    model.compile( loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


    img = image.load_img('./data/c0.jpg', target_size=image_size)
    img_array = image.img_to_array(img)
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    print('pred',pred)
    #给出类别预测：0或者1
    result = model.predict_classes(x)   

    print('\n result',result[0])


if __name__ == '__main__':

    main()

    # pred_data()

    # load_data()
