# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from keras import utils, callbacks
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, Concatenate, Dot,Reshape,Merge
from keras.optimizers import Adam, SGD, RMSprop,Adamax
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer,LabelEncoder
from sklearn.svm import SVR

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def load_data():
    # 加载items数据
    header = ['id', 'price', 'discount', 'platdiscount','terminal','type','orderdate','name', 'store', 'c3', 'brand', 'qty']
    item_df = pd.read_csv('./data/products/items.csv', sep=',', names=header)
    X_train, X_test, y_train, y_test = train_test_split( X_item, y_item, test_size=0.2)
    print(item_df.shape)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    print(train.shape, test.shape)

    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

    for f in test.columns:
        if test[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(test[f].values))
            test[f] = lbl.transform(list(test[f].values))

    return train, test


def feature_selection():
    """
    特征选择
    """
    x, y = load_data()
    params = {
        # 节点的最少特征数 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
        'min_child_weight': 100,
        'eta': 0.02,        # 如同学习率 [默认0.3]
        'colsample_bytree': 0.7,   # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
        # 这个值为树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。 需要使用CV函数来进行调优。 典型值：3-10
        'max_depth': 12,
        'subsample': 0.8,   # 采样训练数据，设置为0.7
        'alpha': 1,         # L1正则化项 可以应用在很高维度的情况下，使得算法的速度更快。
        'gamma': 1,         # Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。
        'silent': 1,        # 0 打印正在运行的消息，1表示静默模式。
        'verbose_eval': True,
        'seed': 12
    }
    xgtrain = xgb.DMatrix(x, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=10)
    features = [x for x in x.columns if x not in ['id', 'qty']]

    create_feature_map(features)
    # 获得每个特征的重要性
    importance = bst.get_fscore(fmap='./data/products/xgb.fmap')
    print("特征数量", len(importance))
    # 重要性排序
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=["feature", "fscore"])
    df["fscore"] = df["fscore"] / df['fscore'].sum()
    print(df)
    label = df['feature'].T.values
    xtop = df['fscore'].T.values
    idx = np.arange(len(xtop))
    fig = plt.figure(figsize=(12, 6))
    plt.barh(idx, xtop, alpha=0.8)
    plt.yticks(idx, label,)
    plt.grid(axis='x')              # 显示网格
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.title('XGBoost 特征选择图示')
    plt.show()


def create_feature_map(features):
    outfile = open('./data/products/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def train():
    X_train, y = load_data()
    print(X_train.shape,y.shape)

    model = Sequential()
    model.add(Dense(128, input_dim=7, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=256, batch_size=256, validation_split=0.2)


if __name__ == '__main__':
    
    # load_data()

    # feature_selection()

    # header = ['id', 'price', 'discount', 'platdiscount','terminal','type','orderdate','name', 'store', 'c3', 'brand', 'qty']
    # item_df = pd.read_csv('./data/products/items.csv', sep=',', names=header)
    # X_item = item_df.drop(['id', 'qty'],axis=1)
    # y_item = item_df['qty']
    # X_train, X_test, y_train, y_test = train_test_split( X_item, y_item, test_size=0.2)
    # print(item_df.shape)
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)
    # train = pd.concat([X_train, y_train], axis=1)
    # test = pd.concat([X_test, y_test], axis=1)
    # print(train.shape, test.shape)

    # for f in train.columns:
    #     if train[f].dtype == 'object':
    #         lbl = LabelEncoder()
    #         lbl.fit(list(train[f].values))
    #         train[f] = lbl.transform(list(train[f].values))

    # for f in test.columns:
    #     if test[f].dtype == 'object':
    #         lbl = LabelEncoder()
    #         lbl.fit(list(test[f].values))
    #         test[f] = lbl.transform(list(test[f].values))

    # return train, test
 