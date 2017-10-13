# -*- coding:utf-8 -*-
# 放假预测，回归任务
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
np.random.seed(seed)


def main():
    house_df = pd.read_csv('./data/housing.csv', sep='\s+', header=None)
    hose_set = house_df.values
    # print(hose_set)
    x = hose_set[:, 0:13]
    y = hose_set[:, 13]
    # print(y)

    # tbcallback=callbacks.TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True, write_images=True)
    estimators = []
    estimators.append(('mlp', KerasRegressor(build_fn=build_model, epochs=512, batch_size=32, verbose=1)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)

    # results = cross_val_score(estimator, x, y, cv=kfold)
    scores = cross_val_score(pipeline, x, y, cv=kfold)
    print('\n')
    print("Results: %.2f (%.2f) MSE" % (scores.mean(), scores.std()))


def build_model():
    model = Sequential()
    model.add(Dense(32, input_dim=13,kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal',activation='relu'))
    model.add(Dense(8, kernel_initializer='normal',activation='linear'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='rmsprop', loss='mae',  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    main()
