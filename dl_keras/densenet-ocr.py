# -*- coding: utf-8 -*-
import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image 
import json
import threading

from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Masking,Lambda,Permute
from keras.layers import Input,Dense,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU,LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam,SGD,Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.utils import plot_model
from matplotlib import pyplot as plt
import tensorflow as tf  


from imp import reload 
import densenet
reload(densenet)

def get_session(gpu_fraction=0.6):  
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''  
  
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)  
  
    if num_threads:  
        return tf.Session(config=tf.ConfigProto(  
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))  
    else:  
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
  
K.set_session(get_session()) 

def ctc_lambda_func(args):
    y_pred,labels,input_length,label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

char=''
with open(r'./data/img_orc/char_std_5990.txt',encoding='utf-8') as f:
      for ch in f.readlines():
            ch = ch.strip('\r\n')
            char=char+ch
char =char[1:]+'卍'
print('nclass:',len(char))

id_to_char = {i:j for i,j in enumerate(char)}
print(id_to_char[5988])

maxlabellength = 20
img_h = 32
img_w = 280
nclass = len(char)
rnnunit=256
batch_size =64


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if(self.index+batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index+batchsize)-self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)

        else:
            r_n = self.range[self.index:self.index+batchsize]
            self.index = self.index+batchsize
        return r_n
    

def readtrainfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip('\r\n'))
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


def gen3(trainfile, batchsize=64, maxlabellength=10, imagesize=(32, 280)):
    """
    图片生成器
    """
    image_label = readtrainfile(trainfile)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength])*10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    print('图片总量', len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(j).convert('L')
            img = np.array(img1, 'f')/255.0-0.5

            x[i] = np.expand_dims(img, axis=2)
            str = image_label[j]
            label_length[i] = len(str)
            if(len(str) <= 0):
                print("len<0", j)
            input_length[i] = imagesize[1]//8
            labels[i, :len(str)] = [int(i)-1 for i in str]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


input = Input(shape=(img_h, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, nclass)

basemodel = Model(inputs=input, outputs=y_pred)
basemodel.summary()

labels = Input(name='the_labels', shape=[maxlabellength], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
adam = Adam()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=adam, metrics=['accuracy'])


checkpoint = ModelCheckpoint(r'./models/img_orc/weights-densent-{epoch:02d}.hdf5',save_weights_only=True)
earlystop = EarlyStopping(patience=10)
tensorboard = TensorBoard(r'./logs/img_orc/tflog-densent',write_graph=True)

print('-----------begin fit-----------')
cc1 = gen3(r'./data/img_orc/train.txt', batchsize=batch_size,maxlabellength=maxlabellength, imagesize=(img_h, img_w))
cc2 = gen3(r'./data/img_orc/test.txt', batchsize=batch_size,maxlabellength=maxlabellength, imagesize=(img_h, img_w))

res = model.fit_generator(cc1,
                          steps_per_epoch=3279601 // batch_size,
                          epochs=100,
                          validation_data=cc2,
                          validation_steps=364400 // batch_size,
                          callbacks=[earlystop, checkpoint, tensorboard],
                          verbose=1
                          )
