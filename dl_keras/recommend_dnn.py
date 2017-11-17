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


k = 256
n_users=943
n_movies=1682
nbatch_size=256
nepochs=64


def load_data():
    user_header = ['user_id', 'age', 'gender', 'occupation']
    user_set = pd.read_csv('./data/ml-100k/u.user', sep='|', names=user_header, usecols=[0, 1, 2, 3], encoding='utf-8')
    
    user_id = pd.get_dummies(user_set['user_id'], prefix='user_id')
    age = pd.get_dummies(user_set['age'], prefix='age')
    gender = pd.get_dummies(user_set['gender'], prefix='gender')
    occ = pd.get_dummies(user_set['occupation'], prefix='occupation')
    user = pd.concat([gender, age, occ], axis=1)
    
    print(user.shape)

    movie_header = ['movie_id', 'title','release_date', 'video_rel_date']
    movie_set = pd.read_csv('./data/ml-100k/u.item.txt', sep='|', names=movie_header, usecols=[0, 1, 2, 3], encoding='utf-8')
    
    title = pd.get_dummies(movie_set['title'], prefix='title')
    r_date = pd.get_dummies(movie_set['release_date'], prefix='release_date')
    v_rel_date = pd.get_dummies(movie_set['video_rel_date'], prefix='video_rel_date')
    movie = pd.concat([title, r_date, v_rel_date], axis=1)

    print(movie.shape)

    rating_header = ['user_id', 'movie_id', 'rating', 'timestamp']
    rating_set = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_header, encoding='utf-8')

    return user, movie, rating_set

def rebuild_data():
    users, movies, ratings = load_data()
    r_users = ratings['user_id'].values
    r_movies = ratings['movie_id'].values
    print(len(r_users),len(r_movies))

    l_user=[]
    for uid in r_users:
        l_user.append(users[(users.user_id == uid)].values[0])
    user_header = ['user_id', 'age', 'gender', 'occupation']
    df_user = pd.DataFrame(data=l_user,columns=user_header)  
    print(df_user)
    df_user.to_csv('./data/ml-100k/user.csv',sep=",",encoding='utf-8')

    l_movies=[]
    for mid in r_movies:
        l_movies.append(movies[(movies.movie_id == mid)].values[0])
    movie_header = ['movie_id', 'title','release_date', 'video_rel_date', 'URL']
    df_movie = pd.DataFrame(data=l_movies,columns=movie_header)  
    print(df_movie)
    df_movie.to_csv('./data/ml-100k/movie.csv',sep=",",encoding='utf-8')


def build_model():

    input_tensor1 = Input(shape=(1, ))
    print("input_tensor1:", input_tensor1.shape)
    first_input = Embedding(100000 + 1, k, input_length=k)(input_tensor1)
    print("first_input:", first_input.shape)

    input_tensor2 = Input(shape=(1, ))
    print("input_tensor2:", input_tensor2.shape)
    second_input = Embedding(100000 + 1, k, input_length=k)(input_tensor2)
    print("second_input:", second_input.shape)

    merged_tensor = K.layers.concatenate([first_input, second_input])
    print("merge:",merged_tensor.shape)

    predictions = Dense(k, activation='relu')(merged_tensor)
    predictions = Dropout(0.3)(predictions)
    predictions = Dense(32, activation='relu')(predictions)
    predictions = Dropout(0.3)(predictions)
    predictions = Dense(8, activation='relu')(predictions)
    predictions = Dropout(0.3)(predictions)
    print("dense_4",predictions.shape)
    predictions = Dense(1, activation='linear')(predictions)

    model = Model(inputs=[input_tensor1, input_tensor2], outputs=predictions)
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    return model



def test_dnn():

    rating_header = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_header, encoding='utf-8')
    ratings.head()

    users = ratings.user_id.unique()
    movies = ratings.movie_id.unique()

    userid2idx = {o: i for i, o in enumerate(users)}
    movieid2idx = {o: i for i, o in enumerate(movies)}
    print(len(userid2idx),len(movieid2idx))

    ratings.movie_id = ratings.movie_id.apply(lambda x: movieid2idx[x])
    ratings.user_id = ratings.user_id.apply(lambda x: userid2idx[x])

    user_min, user_max, movie_min, movie_max = (ratings.user_id.min(), ratings.user_id.max(),
                                                ratings.movie_id.min(), ratings.movie_id.max())

    n_users = ratings.user_id.nunique()
    n_movies = ratings.movie_id.nunique()
    n_factors = 50
    np.random.seed = 42
    print(n_users, n_movies)

    msk = np.random.rand(len(ratings)) < 0.8
    trn = ratings[msk]
    val = ratings[~msk]

    g = ratings.groupby('user_id')['rating'].count()
    topUsers = g.sort_values(ascending=False)[:15]
    g = ratings.groupby('movie_id')['rating'].count()
    topMovies = g.sort_values(ascending=False)[:15]
    top_r = ratings.join(topUsers, rsuffix='_r', how='inner', on='user_id')
    top_r = top_r.join(topMovies, rsuffix='_r', how='inner', on='movie_id')

    def embedding_input(name, n_in, n_out, reg):
        inp = Input(shape=(1,), dtype='int64', name=name)
        return inp, Embedding(n_in, n_out, input_length=1)(inp)

    def create_bias(inp, n_in):
        x = Embedding(n_in, 1, input_length=1)(inp)
        return Flatten()(x)

    user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
    movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)

    # ub = create_bias(user_in, n_users)
    # mb = create_bias(movie_in, n_movies)
    
    # # x = merge([u, m], mode='dot')
    # x = K.layers.dot([u, m], axes=-1)
    # print(x.shape)
    # x = Flatten()(x)
    # x = K.layers.add([x, ub])
    # x = K.layers.add([x, mb])
    # # x = merge([x, ub], mode='sum')
    # # x = merge([x, mb], mode='sum')
    # print("merged",x.shape)

    x = K.layers.concatenate([u, m], axis=-1)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1)(x)

    model = Model([user_in, movie_in], x)
    model.compile(Adam(0.001), loss='mse',metrics=['accuracy'])
    utils.plot_model(model,to_file='./models/test_rnn_model.png',show_shapes=True)
    
    callTB = callbacks.TensorBoard(log_dir='./logs/5')
    model.fit([trn.user_id, trn.movie_id], trn.rating, batch_size=128, epochs=16,callbacks=[callTB], 
              validation_data=([val.user_id, val.movie_id], val.rating))
 
    x_test=[np.array([253]), np.array([465])]
    r=model.predict(x_test)
    print("预测结果：",r)



if __name__ == '__main__':
    
    # user, movie, rating_set=  load_data()
    # print(user.shape,movie.shape)

    # rebuild_data()
    
    test_dnn()
