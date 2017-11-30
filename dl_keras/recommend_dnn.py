import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from keras import utils,callbacks
from keras.models import Sequential,Model
from keras.layers import Input, Dense, LSTM,Flatten, Embedding,Reshape,Dropout,merge
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import keras as K
import scipy as spy
import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

# 定义 placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')

k = 256
n_users = 943
n_movies = 1682


def load_data():
    user_header = ['user_id','gender', 'age',  'job']
    user_set = pd.read_csv('./data/ml-1m/users.dat', sep='::', names=user_header, usecols=[0, 1, 2, 3], encoding='utf-8')
    
    # user_id = pd.get_dummies(user_set['user_id'], prefix='user_id')
    # gender = pd.get_dummies(user_set['gender'], prefix='gender')
    # age = pd.get_dummies(user_set['age'], prefix='age')
    # job = pd.get_dummies(user_set['job'], prefix='job')
    # user = pd.concat([gender, age, job], axis=1)

    movie_header = ['movie_id', 'title','category']
    movie_set = pd.read_csv('./data/ml-100k/u.item.txt', sep='::', names=movie_header, usecols=[0, 1, 2], encoding='utf-8')
    
    # movie_id = pd.get_dummies(movie_set['movie_id'], prefix='movie_id')
    # title = pd.get_dummies(movie_set['title'], prefix='title')
    # category = pd.get_dummies(movie_set['category'], prefix='category')
    # movie = pd.concat([title, category], axis=1)

    rating_header = ['user_id', 'movie_id', 'rating', 'timestamp']
    rating_set = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_header, encoding='utf-8')

    return user, movie, rating_set


def test_dnn():
    #获取最大ID
    n_users = np.max(ratings['user_id'])
    n_movies = np.max(ratings['movie_id'])
    #获取用户和电影列表
    users = ratings['user_id'].values
    movies = ratings['movie_id'].values

    #训练数据集
    X_train = [users, movies]
    #标签
    y_train = ratings['rating'].values

    #先构建2个小的神经网络
    input_1 = Input(shape=(1,),name='main_input_1')
    input_1_emb = Embedding(n_users + 1, k, input_length = 1)(input_1)
    model_1 = Reshape((k,))(input_1_emb)

    input_2 = Input(shape=(1,),name='main_input_2')
    input_2_emb = Embedding(n_movies + 1, k, input_length = 1)(input_2)
    model_2 = Reshape((k,))(input_2_emb)
    #合并两个神经网络
    input_x = dot(inputs=[model_1, model_2],axes=1)

    model = Model(inputs = [input_1,input_2], outputs = input_x)
    model.compile(loss='mse', optimizer='adam')
    #这是我之前训练好的权重
    model.load_weights("movie.h5")



def embedding_input(name, n_in, n_out, emb_size):
    inp = Input(shape=(n_in,), dtype='int64', name=name)
    emb = Embedding(n_in, n_out, input_length=emb_size)(inp)
    fc = Dense(emb_size)(emb)
    return inp, fc


def get_usr_combined_features():
    n_factors = 32
    user_header = ['user_id','gender', 'age',  'job']
    user_set = pd.read_csv('./data/ml-1m/users.dat', sep='::', names=user_header, usecols=[0, 1, 2, 3], engine = 'python')
    user_id = pd.get_dummies(user_set['user_id'], prefix='user_id')
    gender = pd.get_dummies(user_set['gender'], prefix='gender')
    age = pd.get_dummies(user_set['age'], prefix='age')
    job = pd.get_dummies(user_set['job'], prefix='job')
    users = pd.concat([user_id,gender, age, job], axis=1)
    print(users.shape)

    # max_uid = np.max(user_set.user_id.values) + 1
    # max_gender = np.max(gender.values) + 1
    # max_age = np.max(user_set.age.values) + 1
    # max_job = np.max(user_set.job.values) + 1
    # input_uid, uid_fc = embedding_input('user_id', max_uid, n_factors,256)    
    # input_gender, gender_fc = embedding_input('gender_id', max_gender, n_factors,256)    
    # input_age, age_fc = embedding_input('age_id', max_age, n_factors,256)    
    # input_job, job_fc = embedding_input('job_id', max_job, n_factors,256) 

    inp_user, user_fc = embedding_input('user', 6070, n_factors,256)    
    print("构建user网络......")
    # inp_usr = K.layers.concatenate([input_uid, input_gender, input_age, input_job],axis=-1)
    # ux = K.layers.concatenate([uid_fc, gender_fc, age_fc, job_fc], axis=-1)
    ux = Flatten()(user_fc)
    ux = Dropout(0.2)(ux)
    user_x = Dense(200, activation='tanh')(ux)
    usr_model = Model(inp_user,user_x)

    return users, inp_user,usr_model


def get_mov_combined_features():
    n_factors = 32
    movie_header = ['movie_id', 'title','category']
    movie_set = pd.read_csv('./data/ml-1m/movies.dat', sep='::', names=movie_header, usecols=[0, 1, 2], engine = 'python')
    movie_id = pd.get_dummies(movie_set['movie_id'], prefix='movie_id')
    title = pd.get_dummies(movie_set['title'], prefix='title')
    category = pd.get_dummies(movie_set['category'], prefix='category')
    movies = pd.concat([movie_id,title, category], axis=1)
    print(movies.shape)
    # max_mid = np.max(movie_set.movie_id.values) + 1
    # max_title = len(title.values)+1
    # max_category = len(category.values)+1
    # print(max_mid,max_title,max_category)
    # input_mid, mid_fc = embedding_input('movie_id', max_mid, n_factors,256)    
    # input_title, title_fc = embedding_input('title_id', max_title, n_factors,256)    
    # input_category, category_fc = embedding_input('category_id', max_category, n_factors,256)  

    inp_mov, user_fc = embedding_input('movie', 8067, n_factors,256)      

    print("构建movie网络......")
    # print(inp_mov)
    # mx = K.layers.concatenate([mid_fc, title_fc, category_fc], axis=-1)
    mx = Flatten()(user_fc)
    mx = Dropout(0.2)(mx)
    movie_x = Dense(200, activation='tanh')(mx)
    mov_model = Model(inp_mov,movie_x)

    return movies,inp_mov,mov_model

def main():
    r_header = ['user_id', 'movie_id', 'rating']
    rating_set = pd.read_csv('./data/ml-1m/ratings.dat',sep='::', names=r_header, engine='python')

    users, inp_usr, usr_model = get_usr_combined_features()
    movies, inp_mov, mov_model = get_mov_combined_features()
    # #训练数据集
    x_train = [users, movies]
    y = rating_set.rating.values

    input_x =K.layers.concatenate(inputs=[usr_model, mov_model],axis=-1)
    model = Model([inp_usr, inp_mov],input_x)
    model.compile(Adam(0.001), loss='mse',metrics=['accuracy'])
    utils.plot_model(model,to_file='./models/recomm_dnn_model.png',show_shapes=True)
    
    print("开始训练......")
    callTB = callbacks.TensorBoard(log_dir='./logs/dnn-1')
    model.fit(x_train, y, batch_size=128, epochs=16, callbacks=[callTB], validation_split=0.2)


if __name__ == '__main__':
    
    # get_usr_combined_features()

    # get_mov_combined_features()

    main()

