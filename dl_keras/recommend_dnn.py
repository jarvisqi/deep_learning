# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from keras import utils, callbacks
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, Concatenate, Dot,Reshape,Merge
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop,Adamax
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing 
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso 

feature_dim = 30
nbatch_size = 512

def rebuild_data():
    """
    清洗选择特征数据
    """
    user_header = ['user_id','gender', 'age',  'job']
    user_df = pd.read_csv('./data/ml-1m/users.dat', sep='::', names=user_header, usecols=[0, 1, 2, 3], engine = 'python')
    user_df.set_index(['user_id'], inplace = False) 

    movie_header = ['movie_id', 'title','category']
    movie_df = pd.read_csv('./data/ml-1m/movies.dat', sep='::', names=movie_header, usecols=[0, 1, 2], engine = 'python')
    movie_df.set_index(['movie_id'], inplace = False) 

    rating_header = ['user_id', 'movie_id', 'rating', 'timestamp']
    rating_df = pd.read_csv('./data/ml-1m/ratings.dat',sep='::', names=rating_header, engine = 'python')[:100000]

    rating_user = [user_df[user_df['user_id'] == mid].values[0] for uid, mid, r, _ in rating_df.values]
    rating_movie = [movie_df[movie_df['movie_id'] == mid].values[0] for uid, mid, r, _ in rating_df.values]

    user_df = pd.DataFrame(rating_user, index=None, columns=['user_id', 'gender', 'age',  'job'])
    movie_df = pd.DataFrame(rating_movie, index=None, columns=['movie_id', 'title', 'category'])
    rating_df = rating_df.rating
    pd.to_pickle(user_df, './data/ml-1m/user_pick')
    pd.to_pickle(movie_df, './data/ml-1m/movie_pick')
    pd.to_pickle(rating_df, './data/ml-1m/rating_pick')

    print(user_df.shape,movie_df.shape,rating_df.shape)


def load_data():
    """
    加载数据
    """
    user_df= pd.read_pickle('./data/ml-1m/user_pick')
    movie_df= pd.read_pickle('./data/ml-1m/movie_pick')
    rating_df= pd.read_pickle('./data/ml-1m/rating_pick')

    user_id = pd.get_dummies(user_df['user_id'], prefix='user_id')
    gender = pd.get_dummies(user_df['gender'], prefix='gender')
    age = pd.get_dummies(user_df['age'], prefix='age')
    job = pd.get_dummies(user_df['job'], prefix='job')
    users_df = pd.concat([gender, age, job], axis=1)

    movie_id = pd.get_dummies(movie_df['movie_id'], prefix='movie_id')
    title = pd.get_dummies(movie_df['title'], prefix='title')
    category = pd.get_dummies(movie_df['category'], prefix='category')
    movies_df = pd.concat([title, category], axis=1)

    return users_df.values, movies_df.values,rating_df.values


def test_dnn():
    k=128
    rating_header = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('./data/ml-1m/ratings.dat',sep='::', names=rating_header, engine = 'python')[:300000]
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
    model1 = Sequential()
    model1.add(Embedding(n_users + 1, k, input_length = 1))
    model1.add(Reshape((k,)))

    model2 = Sequential()
    model2.add(Embedding(n_movies + 1, k, input_length = 1))
    model2.add(Reshape((k,)))

    model = Sequential()
    model.add(Merge([model1, model2], mode = 'concat'))
    model.add(Dropout(0.2))
    model.add(Dense(k, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(k/4), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(k/16), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=64, batch_size=512,validation_split=0.2)

    sumt = 0
    for i in range(ratings.shape[0]):
        sumt += (ratings['rating'][i] - model.predict([np.array([ratings['user_id'][i]]), np.array([ratings['movie_id'][i]])])) ** 2
    mse = math.sqrt(sumt/ratings.shape[0])
    print(mse)


def build_model(x_train,y_train):
    """
    构建网络，训练模型
    """

    print("build network")
    usr_input = Input(shape=(feature_dim,))
    usr_x = Embedding(x_train[0].shape[0] + 1, 256, input_length=30)(usr_input)
    print("user_embedding_x:", usr_x.shape)
    usr_x = Flatten()(usr_x)
    usr_x = Dense(128, activation='relu')(usr_x)
    print("user_dense_x:", usr_x.shape)

    mov_input = Input(shape=(feature_dim,))
    mov_x = Embedding(x_train[0].shape[0] + 1, 256, input_length=30)(mov_input)
    print("movie_embedding_x:", mov_x.shape)
    mov_x = Flatten()(mov_x)
    mov_x = Dense(128, activation='relu')(mov_x)
    print("movie_dense_x:", mov_x.shape)

    concat_tensor = Concatenate()([usr_x, mov_x])
    print("concat_tensor:", concat_tensor.shape)
    x_tensor = Dense(64, activation='relu')(x_tensor)
    x_tensor = Dropout(0.5)(x_tensor)
    x_tensor = Dense(32, activation='relu')(x_tensor)
    x_tensor = Dropout(0.3)(x_tensor)
    x_output = Dense(1, activation='linear')(x_tensor)

    print("Model:", usr_input.shape, mov_input.shape, "output_x:", x_output.shape)
    model = Model([usr_input, mov_input], x_output)
    sgd = Adam(lr=0.002)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model_png='./models/dnn_recomm_model.png'
    # 显示网络结构 
    if not os.path.exists(model_png):
        utils.plot_model(model,to_file='./models/dnn_recomm_model.png')
    callTB = callbacks.TensorBoard(log_dir='./logs/dnn_merge-2')
    print("training model")
    best_model = callbacks.ModelCheckpoint("./models/dnn_recommend_full.h5", monitor='val_loss', verbose=0, save_best_only=True)
    model.fit(x_train, y_train, epochs=64, batch_size=512,callbacks=[callTB, best_model], validation_split=0.2)


def build_SqModel(x_train,y_train):
    
    model = Sequential()
    model.add(Embedding(100000 + 1, 128, input_length=20))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy' ,metrics=['accuracy'])
    callTB = K.callbacks.TensorBoard(log_dir='./logs/dnn_merge-1')
    model.fit(x_train, y_train, epochs=16, batch_size=32,callbacks=[callTB],validation_split=0.2)



def sign_model():
    users, movies, ratings = load_data()
    x_train = np.concatenate([users, movies], axis=1)
    y_train = ratings

    # 特征选择  用来计算特征的重要程度，因此能用来去除不相关的特征
    clf = RandomForestClassifier()
    clf = clf.fit(x_train, ratings)
    ft_model = SelectFromModel(clf, prefit=True)
    x_train = ft_model.transform(x_train)
    print(x_train.shape, y_train.shape)
    dim=x_train.shape[1]

    x_input = Input(shape=(dim,))
    x_tensor = Dense(128, activation='relu')(x_input)
    x_tensor = Dropout(0.5)(x_tensor)
    x_tensor = Dense(32, activation='relu')(x_tensor)
    x_tensor = Dropout(0.3)(x_tensor)
    x_tensor = Dense(8, activation='relu')(x_tensor)
    x_tensor = Dropout(0.3)(x_tensor)
    x_output = Dense(1, activation='linear')(x_tensor)

    model = Model(x_input, x_output)
    sgd = Adam(lr=0.001)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=64, batch_size=nbatch_size,callbacks=None, validation_split=0.2)
    model.save("./models/dnn_sign_model.h5")


def main():
    print("load data")
    users, movies, ratings = load_data()
    print("origin dim", users.shape, movies.shape)

    print("PCA")
    usr_pca = PCA(n_components=feature_dim).fit_transform(users)
    mov_pca = PCA(n_components=feature_dim).fit_transform(movies)
    print("user_pca dim:", usr_pca.shape, "movie_pca dim:", mov_pca.shape)

    x_train = [usr_pca, mov_pca]
    y_train = ratings

    build_model(x_train, y_train)

    #  x_train = usr_pca
    #  build_SqModel(x_train,y_train)


if __name__ == '__main__':

    # test_dnn()

    # main()

    # load_data()     # (100000, 30) (100000, 3763) (100000,)

    # rebuild_data()

    sign_model()

