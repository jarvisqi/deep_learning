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
    df_user.to_csv('../data/ml-100k/user.csv',sep=",",encoding='utf-8')

    l_movies=[]
    for mid in r_movies:
        l_movies.append(movies[(movies.movie_id == mid)].values[0])
    movie_header = ['movie_id', 'title','release_date', 'video_rel_date', 'URL']
    df_movie = pd.DataFrame(data=l_movies,columns=movie_header)  
    print(df_movie)
    df_movie.to_csv('../data/ml-100k/movie.csv',sep=",",encoding='utf-8')


def build_model():

    input_tensor1 = Input(shape=(100000, 5))
    print("input_tensor1:", input_tensor1.shape)
    first_input = Embedding(100000 + 1, k, input_length=k)(input_tensor1)
    print("first_input:", first_input.shape)

    input_tensor2 = Input(shape=(100000, 5))
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


def train_model():
    users, movies, ratings = load_data()
    print(users.shape,movies.shape)
    r_users = ratings['user_id'].values
    r_movies = ratings['movie_id'].values
    print(r_users.shape,r_movies.shape)

    user_set = pd.read_csv('./data/ml-100k/user.csv', sep=',', encoding='utf-8')
    occupation = [name for name, group in user_set.groupby(['occupation'])]
    occ_index = [occupation.index(i) for i in occupation]
    user_set = user_set.replace(['F', 'M'], [0, 1], inplace=False)
    user_set = user_set.replace(occupation, occ_index, inplace=False)
    # # n=user_set.loc[0].values
    # # movies=np.tile(r_movies,(5,1)).T
    # frame=user_set.drop_duplicates(['user_id'])  
    # print(frame.values.shape)
    users = user_set.values

    # movies=movies.reshape(-1)
    # users=users.reshape(-1)
    label = ratings['rating'].values
    # label=np.tile(label,(5,1)).T.reshape(-1)
    # # users 和 movies必须是 相同的张量
    
    # x_train = [users, movies]
    print(users.shape,r_movies.shape,label.shape)
    # users = np.expand_dims(users, axis=0)
    x_train = [np.array(users), np.array(users)]
    y_train = label

    model = build_model()
    plotpath='./models/recommand_rnn_model.png'
    if not os.path.exists(plotpath):
        utils.plot_model(model,to_file=plotpath,show_shapes=True)

    print("training model")
    callTB = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True)
    best_model = ModelCheckpoint("./models/recommand_rnn_full.h5", monitor='val_loss', verbose=0, save_best_only=True)
    print("x_train",x_train[0].shape,x_train[1].shape)
    model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs,verbose=1, callbacks=[callTB,best_model], validation_split=0.15)


def test_conc():
    users, movies, ratings = load_data()

    input_tensor1 = Input(shape=(k,))
    print("input_tensor1:", input_tensor1.shape)
    first_input = Embedding(100000 + 1, k, input_length=1)(input_tensor1)
    print("first_input:", first_input.shape)

    input_tensor2 = Input(shape=(k,))
    print("input_tensor2:", input_tensor2.shape)
    second_input = Embedding(100000 + 1, k, input_length=1)(input_tensor2)
    print("second_input:", second_input.shape)

    merged_tensor = K.layers.concatenate([first_input, second_input])
    print("merge:",merged_tensor.shape)

    model = Model(inputs=[input_tensor1], outputs=first_input)
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    label = ratings['rating'].values
    print(label.shape)
    # label=np.tile(label,(5,1)).T.reshape(-1)
    ua=np.zeros((1,20788))
    ub=np.expand_dims(users.values.reshape(-1),axis=0)
    users=np.column_stack((ua,ub))

    ma=np.zeros((1,93272))
    mb=np.expand_dims(movies.values.reshape(-1),axis=0)
    movies=np.column_stack((ma,mb))

    print(users.shape)
    print(movies.shape)

    x_train = [users, movies]
    y_train = label

    model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs,verbose=1, validation_split=0.15)


def test_rnn():

    rating_header = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_header, encoding='utf-8')
    ratings.head()

    users = ratings.user_id.unique()
    movies = ratings.movie_id.unique()

    userid2idx = {o: i for i, o in enumerate(users)}
    movieid2idx = {o: i for i, o in enumerate(movies)}
    print(len(userid2idx),len(movieid2idx))

    ratings.movieId = ratings.movie_id.apply(lambda x: movieid2idx[x])
    ratings.userId = ratings.user_id.apply(lambda x: userid2idx[x])

    user_min, user_max, movie_min, movie_max = (ratings.userId.min(), ratings.userId.max(),
                                                ratings.movieId.min(), ratings.movieId.max())

    n_users = ratings.userId.nunique()
    n_movies = ratings.movieId.nunique()
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

    ub = create_bias(user_in, n_users)
    mb = create_bias(movie_in, n_movies)
    
    # x = merge([u, m], mode='dot')
    x = K.layers.dot([u, m], axes=-1)
    print(x.shape)
    x = Flatten()(x)
    x = K.layers.add([x, ub])
    x = K.layers.add([x, mb])
    # x = merge([x, ub], mode='sum')
    # x = merge([x, mb], mode='sum')
    print(x.shape)

    model = Model([user_in, movie_in], x)
    model.compile(Adam(0.001), loss='mse',metrics=['accuracy'])
    utils.plot_model(model,to_file='./models/test_rnn_model.png',show_shapes=True)
    
    callTB = callbacks.TensorBoard(log_dir='./logs/2')
    model.fit([trn.user_id, trn.movie_id], trn.rating, batch_size=128, epochs=8,callbacks=[callTB], 
              validation_data=([val.user_id, val.movie_id], val.rating))
 
    x_test=[np.array([253]), np.array([465])]
    r=model.predict(x_test)
    print(r)



if __name__ == '__main__':
    
    # test_conc()

    # train_model()

    # load_data()

    # rebuild_data()
    
    test_rnn()
