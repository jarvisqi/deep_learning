# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import jieba
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras import callbacks, utils
from keras.preprocessing import text,sequence
from keras.models import Sequential,model_from_yaml, load_model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam



np.random.seed(7)
# set parameters:
MAX_SEQUENCE_LENGTH = 100   # 每条语句最大长度
EMBEDDING_DIM = 200         # 词向量空间维度
window_size = 7
batch_size = 128
cpu_count = multiprocessing.cpu_count()


def load_data():
    """
    加载数据
    """
    neg = pd.read_excel("./data/text/neg.xls", header=None, index=None)
    pos = pd.read_excel("./data/text/pos.xls", header=None, index=None)

    data = np.concatenate((pos[0], neg[0]))
    labels = np.concatenate((np.ones(len(pos), dtype=int),np.zeros(len(neg), dtype=int)))
    return data, labels


def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined = parse_dataset(combined)
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        combined = sequence.pad_sequences(combined, maxlen=MAX_SEQUENCE_LENGTH)
        return w2indx, w2vec, combined
    else:
        print('No data')


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(text):

    model = word2vec.Word2Vec(size=EMBEDDING_DIM, min_count=10, window=window_size, workers=cpu_count,iter=1)
    model.build_vocab(text)
    model.train(text, total_examples=model.corpus_count, epochs=model.iter)
    model.save('./models/Word2vec_model.model')
    index_dict, word_vectors, text = create_dictionaries(model=model, combined=text)
    return index_dict, word_vectors, text



# 定义网络结构
def train_model(input_dim,x_train, y_train, x_test, y_test):
    print(input_dim)    
    print('设计模型 Model...')

    model = Sequential()

    model.add(Embedding(input_dim,EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1,activation="sigmoid"))

    print('编译模型...')   # 使用 adam优化
    sgd = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    tbCallBack= callbacks.TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True, write_images=True)

    print("训练...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=3,verbose=1, validation_data=(x_test, y_test),callbacks=[tbCallBack])
    # epochs=16 时精度最高：0.9209,  921837993421 
    print("评估...")
    score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('\nTest score:', score)
    print('Test accuracy:', accuracy)

    yaml_string = model.to_yaml()
    with open('./models/lstm.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./models/lstm.h5')


def train():
    """
    训练模型，并保存

    """
    print('Loading Data...')
    inputTexts, labels = load_data()
    print(inputTexts.shape, labels.shape)

    print('segment...')

    # seg_data = [jieba.lcut(document.replace('\n', ''))for document in inputTexts]
    # print('word2vec...')
    # index_dict, word_vectors, data = word2vec_train(seg_data)
    # n_symbols = len(index_dict) + 1   
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    # print(x_train.shape, y_train.shape)

    # train_model(n_symbols, x_train, y_train, x_test, y_test)

    texts=[]
    for doc in inputTexts:
        seg_doc = jieba.lcut(doc.replace('\n', ''))
        d=merge_doc(seg_doc)
        texts.append(d)
    tokenizer = text.Tokenizer()                            # 分词MAX_NB_WORDS
    tokenizer.fit_on_texts(texts)
    text_sequences = tokenizer.texts_to_sequences(texts)    # 受num_words影响
    word_index = tokenizer.word_index                       # 词_索引
    data = sequence.pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    input_dim=len(word_index) + 1
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    print(x_train.shape, y_train.shape)

    train_model(input_dim, x_train, y_train, x_test, y_test)


def predictData():
    """
    使用模型预测真实数据

    """
    input_texts = ["虽然价格很贵，但是推荐大家购买"]

    texts = [jieba.lcut(document.replace('\n', '')) for document in input_texts]
    word_model = word2vec.Word2Vec.load('./models/Word2vec_model.model')
    w2indx, w2vec, texts = create_dictionaries(word_model, texts)
    # 加载网络结构
    with open('./models/lstm.yaml', 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()
    model = model_from_yaml(loaded_model_yaml)
    # 加载模型权重
    model.load_weights("./models/lstm.h5")
    print("model Loaded")

    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
                  
    utils.plot_model(model,to_file='./models/lstm_model.png')
    # # 预测
    pred_result = model.predict_classes(texts)
    labels = [int(round(x[0])) for x in pred_result]
    label2word = {1: '正面', 0: '负面'}
    for i in range(len(pred_result)):
        print('{} -------- {}'.format(label2word[labels[i]], input_texts[i]))


def merge_doc(x):
    doc=''
    for d in x:
        doc +=d+' '
    return doc

if __name__ == '__main__':
    
    train()

    # predictData()
