import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

np.random.seed(7)
maxlen = 40
step = 3
nbatch_size = 256
sentences = []
next_chars = []


def load_data():

    alltext = open("./data/text/四世同堂.txt", encoding="UTF-8").read()
    sortedcharset = sorted(set(alltext))
    char_index = dict((c, i) for i, c in enumerate(sortedcharset))
    index_char = dict((i, c) for i, c in enumerate(sortedcharset))

    for i in range(0, len(alltext) - maxlen, step):
        sentences.append(alltext[i: i + maxlen])
        next_chars.append(alltext[i + maxlen])

    print('nb sequences:', len(sentences))

    # x = np.zeros((len(sentences), maxlen, len(sortedcharset)), dtype=np.bool)
    # y = np.zeros((len(sentences), len(sortedcharset)), dtype=np.bool)
    # for i, sentence in enumerate(sentences):
    #     if (i % 30000 == 0):
    #         print(i)
    #     for t in range(maxlen):
    #         char = sentence[t]
    #         x[i, t, char_indices[char]] = 1
    #         y[i, char_indices[next_chars[i]]] = 1

    return x, y, sortedcharset

def data_generator(x, y, batch_size):
    if batch_size < 1:
        batch_size = nbatch_size
    number_of_batches = x.shape[0] // batch_size
    counter = 0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)

    #reset generator
    while 1:
        index_batch = shuffle_index[batch_size * counter:batch_size * (counter + 1)]
        x_batch = (x[index_batch, :, :]).astype('float32')
        y_batch = (y[index_batch, :]).astype('float32')
        counter += 1

    yield(np.array(X_batch), y_batch)

    if (counter < number_of_batches):
        np.random.shuffle(shuffle_index)
        counter = 0
    

def build_model(sortedcharset):
    print('Build model...')
    model=Sequential()
    model.add(LSTM(256,input_shape=(maxlen, len(sortedcharset)), recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(len(sortedcharset)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def train_model():
    x, y, sortedcharset = load_data()

    model = build_model(sortedcharset)
    tbCallBack = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    best_model = callbacks.ModelCheckpoint("./models/gen_wrods.h5", monitor='val_loss', verbose=0, save_best_only=True)
    model.fit_generator(data_generator(x, y, nbatch_size), steps_per_epoch=512, epochs=16, verbose=1, callbacks=[tbCallBack, best_model],
                        validation_data=data_generator(x, y, nbatch_size), validation_steps=128)



if __name__ == '__main__':
    train_model()
