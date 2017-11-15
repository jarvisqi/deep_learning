import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM


n_epochs=32
n_batch_size=64
n_hours=3
n_features = 8



def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def load_data():
    data_set = pd.read_csv('./data/text/raw.csv', parse_dates=[
                           ['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    # 删除一列
    data_set.drop('No', axis=1, inplace=True)

    # 指定列名
    data_set.columns = ['pollution', 'dew', 'temp',
                        'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    data_set.index.__name__ = 'date'

    # 用0填充的NA值
    data_set['pollution'].fillna(0, inplace=True)
    # print(data_set)
    data_set = data_set[24:]

    # print(data_set.head(5))

    data_set.to_csv('./data/text/pollution.csv')


def show_pyplot():
    data_set = pd.read_csv('./data/text/pollution.csv', header=0, index_col=0)
    values = data_set.values

    # 显示5年的变化曲线
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(data_set.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()


def load_pollution():
    data_set = pd.read_csv('./data/text/pollution.csv', header=0, index_col=0)
    values = data_set.values

    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    values=values.astype('float32')
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # print(scaled)

    # 构造为监督学习
    reframed = series_to_supervised(scaled,n_hours,1)
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    # print('reframed head \n',reframed.head())

    values=reframed.values
    n_train_hours=365*24
    train=values[:n_train_hours]
    test=values[n_train_hours:,:]
    n_obs = n_hours * n_features
    # 切分数据机集
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape 为3D的张量 [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return train_X, test_X, train_y, test_y,scaler


def build_model(x,y):
    
    model=Sequential()
    model.add(LSTM(64,input_shape=(x,y)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mae',metrics=['accuracy'])

    return model

def train_model():

    train_x, test_x, train_y, test_y, scaler = load_pollution()
    model = build_model(train_x.shape[1], train_x.shape[2])
    history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch_size,
                        verbose=1, validation_data=(test_x, test_y), shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # predict
    yhat = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], n_hours * n_features))
    # print(yhat.shape,test_x[:, -7:].shape)
    inv_yhat = np.concatenate((yhat, test_x[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_x[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    print(n_vars)
    df = pd.DataFrame(data)
    # print(df)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # shift(-2, freq='D') #后移、前移
    for i in range(0, n_out):
        cols.append(df.shift(-1))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # print(cols)
    agg = pd.concat(cols, axis=1)   # 连接2序列，按行的方向
    agg.columns = names
    if dropnan:
    	agg.dropna(inplace=True)

    # print(agg)
    return agg



if __name__ == '__main__':
    
    train_model()

