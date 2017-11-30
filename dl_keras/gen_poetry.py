# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import pandas as pd
import numpy as np
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout,Embedding,GRU,LSTM,BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils, plot_model

nbatch_size = 512
nepochs = 100

def main():
    print('load data')
    with open('./data/text/poetry.txt',mode='r',encoding='utf-8') as f:
        raw_text = f.read()
        lines = raw_text.split("\n")[:-1]
        poem_text = [i.split(':')[1] for i in lines]
        char_list = [re.findall('[\x80-\xff]{3}|[\w\W]', s) for s in poem_text]
    all_words = []
    for i in char_list:
        all_words.extend(i)
    word_dataframe = pd.DataFrame(pd.Series(all_words).value_counts())
    word_dataframe['id'] = list(range(1, len(word_dataframe) + 1))
    word_index_dict = word_dataframe['id'].to_dict()
    pd.to_pickle(word_index_dict,'./data/text/word_index_dict')
    
    index_dict = {}
    for k in word_index_dict:
        index_dict.update({word_index_dict[k]: k})
    seq_len = 2
    dataX = []
    dataY = []
    print("build data")
    for i in range(0, len(all_words) - seq_len, 1):
        seq_in = all_words[i: i + seq_len]
        seq_out = all_words[i + seq_len]
        dataX.append([word_index_dict[x] for x in seq_in])
        dataY.append(word_index_dict[seq_out])
    x = np.array(dataX)
    print(len(dataY))
    y = np_utils.to_categorical(np.array(dataY))
    print(x.shape)
    print(y.shape,y.shape[1])
    
    print("build network")
    model = Sequential()
    model.add(Embedding(len(word_dataframe) + 1, 256))
    model.add(GRU(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(1024,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(y.shape[1],activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.007), metrics=['accuracy'])
    callTB = callbacks.TensorBoard(log_dir='./logs/rnn-5')
    model.fit(x, y, batch_size=nbatch_size, epochs=nepochs,verbose=1, callbacks=[callTB], validation_split=0.15)
    model.save('./models/gen_poetry_full.h5')


def gen_poem(seed_text):
    
    model = load_model('./models/gen_poetry_full.h5')
    word_index_dict = pd.read_pickle(word_index_dict, './data/text/word_index_dict')
    rows = 4
    cols = 6
    chars = re.findall('[\x80-\xff]{3}|[\w\W]', seed_text)
    if len(chars) != seq_len:
        return ""
    arr = [word_index_dict[k] for k in chars]
    for i in range(seq_len, rows * cols):
        if (i+1) % cols == 0:
            if (i+1) / cols == 2 or (i+1) / cols == 4:
                arr.append(2)
            else:
                arr.append(1)
        else:
            proba = model.predict(np.array(arr[-seq_len:]), verbose=0)
            predicted = np.argsort(proba[1])[-5:]
            index = random.randint(0,len(predicted)-1)
            new_char = predicted[index]
            while new_char == 1 or new_char == 2:
                index = random.randint(0,len(predicted)-1)
                new_char = predicted[index]
            arr.append(new_char)
    poem = [index_dict[i] for i in arr]
    return "".join(poem)



if __name__ == '__main__':
    
    main()
