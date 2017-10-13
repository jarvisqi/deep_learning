# -*- coding:utf-8 -*-
# 使用keras深度学习实现回归问题示例
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


def main():
    x = np.linspace(-1, 1, 400)
    np.random.shuffle(x)

    y = 0.5 * x + 2 + np.random.normal(0, 0.05, (400,))
    # plt.scatter(x, y)
    # plt.show()

    # 构建模型
    model = Sequential()
    model.add(Dense(1, input_dim=1))

    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.85)
    print('train.....')
    model.fit(x_train, y_train, batch_size=32, epochs=512, validation_data=(x_test, y_test))

    print('evaluate.....')
    # score, accuracy
    score, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print('score:', accuracy, 'accuracy', accuracy)

    # 获取参数
    w, b = model.layers[0].get_weights()
    print('weights:', w)
    print('biases:', b)


if __name__ == '__main__':
    main()
