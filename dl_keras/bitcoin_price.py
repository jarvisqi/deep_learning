# -*- coding: utf-8 -*-
import datetime
import io
import os
import sys
import time
import urllib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_data():
    bit_url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end={}"
    bitcoin_market_info = pd.read_html(bit_url.format(time.strftime("%Y%m%d")))[0]
    bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info["Date"]))
    bitcoin_market_info.loc[bitcoin_market_info["Volume"] == "-", "Volume"] = 0
    bitcoin_market_info["Volume"] = bitcoin_market_info["Volume"].astype("int64")
    # print(bitcoin_market_info.head())

    eth_url = "https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end={}"
    eth_market_info = pd.read_html(eth_url.format(time.strftime("%Y%m%d")))[0]
    eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info["Date"]))
    # print(eth_market_info.head())

    return bitcoin_market_info, eth_market_info

def get_bitcoinlogs():
        
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    eth_img = urllib.request.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")
    
    image_file = io.BytesIO(bt_img.read())
    bitcoin_im = Image.open(image_file)

    image_file = io.BytesIO(eth_img.read())
    eth_im = Image.open(image_file)

    # plt.suptitle('Multi Image') # 图片名称
    # plt.subplot(1,2,1), plt.title('bitcoin')
    # plt.imshow(bitcoin_im), plt.axis('off')

    # plt.subplot(1,2,2), plt.title('eth')
    # plt.imshow(eth_im), plt.axis('off')

    # plt.show()
    width_eth_im, height_eth_im = eth_im.size
    eth_im = eth_im.resize((int(eth_im.size[0]*0.8), int(eth_im.size[1]*0.8)), Image.ANTIALIAS)

    bitcoin_market_info, eth_market_info = get_data()
    bitcoin_market_info.columns = [bitcoin_market_info.columns[0]]+["bt_"+ i for i in bitcoin_market_info.columns[1:]]
    eth_market_info.columns = [eth_market_info.columns[0]]+["eth_"+ i for i in eth_market_info.columns[1:]]

    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
    ax1.set_ylabel("Closing Price ($)",fontsize=12)
    ax2.set_ylabel("Volume ($ bn)",fontsize=12)
    ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
    ax2.set_yticklabels(range(10))
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
    ax1.plot(bitcoin_market_info['Date'].astype(datetime.datetime),bitcoin_market_info['bt_Open'])
    ax2.bar(bitcoin_market_info['Date'].astype(datetime.datetime).values, bitcoin_market_info['bt_Volume'].values)
    fig.tight_layout()
    fig.figimage(bitcoin_im, 260, 240, zorder=3,alpha=.5)
    plt.show()


    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
    #ax1.set_yscale('log')
    ax1.set_ylabel('Closing Price ($)',fontsize=12)
    ax2.set_ylabel('Volume ($ bn)',fontsize=12)
    ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
    ax2.set_yticklabels(range(10))
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
    ax1.plot(eth_market_info['Date'].astype(datetime.datetime),eth_market_info['eth_Open'])
    ax2.bar(eth_market_info['Date'].astype(datetime.datetime).values, eth_market_info['eth_Volume'].values)
    fig.tight_layout()
    fig.figimage(eth_im, 360, 280, zorder=3, alpha=.6)
    plt.show()

    market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])
    market_info = market_info[market_info['Date']>='2016-01-01']
    for coins in ['bt_', 'eth_']: 
        kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
        market_info = market_info.assign(**kwargs)
    print(market_info.head())

    end_date = '2017-06-01'
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
    ax1.plot(market_info[market_info['Date'] < end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] < end_date]['bt_Close'], 
            color='#B08FC7', label='Training')
    ax1.plot(market_info[market_info['Date'] >= end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] >= end_date]['bt_Close'], 
            color='#8FBAC8', label='Test')
    ax2.plot(market_info[market_info['Date'] < end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] < end_date]['eth_Close'], 
            color='#B08FC7')
    ax2.plot(market_info[market_info['Date'] >= end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] >= end_date]['eth_Close'], color='#8FBAC8')
    ax1.set_xticklabels('')
    ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
    ax2.set_ylabel('Ethereum Price ($)',fontsize=12)
    plt.tight_layout()
    ax1.legend(bbox_to_anchor=(0.03, 1), loc=2,borderaxespad=0., prop={'size': 14})
    fig.figimage(bitcoin_im.resize((int(bitcoin_im.size[0]*0.65), int(bitcoin_im.size[1]*0.65)), Image.ANTIALIAS),
                 360, 420, zorder=3, alpha=.5)
    fig.figimage(eth_im.resize((int(eth_im.size[0]*0.65), int(eth_im.size[1]*0.65)), Image.ANTIALIAS),
                 460, 120, zorder=3, alpha=.5)
    plt.show()

    # trivial lag model: P_t = P_(t-1)
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
    ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(market_info[market_info['Date']>= end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date']>= end_date]['bt_Close'].values, label='Actual')
    ax1.plot(market_info[market_info['Date']>= end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date']>= datetime.datetime.strptime(end_date, '%Y-%m-%d') - 
                        datetime.timedelta(days=1)]['bt_Close'][1:].values, label='Predicted')
    ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    ax1.set_title('Simple Lag Model (Test Set)')
    ax2.set_ylabel('Etherum Price ($)',fontsize=12)
    ax2.plot(market_info[market_info['Date']>= end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date']>= end_date]['eth_Close'].values, label='Actual')
    ax2.plot(market_info[market_info['Date']>= end_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date']>= datetime.datetime.strptime(end_date, '%Y-%m-%d') - 
                        datetime.timedelta(days=1)]['eth_Close'][1:].values, label='Predicted')
    fig.tight_layout()
    plt.show()


class LSTMNet(object):
    """
    定义网络，训练网络
    """

    def __init__(self,opt):
        self.end_date = '2017-06-01'
        self.window_len = 10
        self.loss = opt.loss
        self.dropout = opt.dropout
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.verbose = opt.verbose

    
    def getData(self):
        bitcoin_market_info, eth_market_info = get_data()
        bitcoin_market_info.columns = [bitcoin_market_info.columns[0]]+["bt_"+ i for i in bitcoin_market_info.columns[1:]]
        eth_market_info.columns = [eth_market_info.columns[0]]+["eth_"+ i for i in eth_market_info.columns[1:]]
        
        market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])
        market_info = market_info[market_info['Date']>='2016-01-01']
        for coins in ['bt_', 'eth_']: 
            kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
            market_info = market_info.assign(**kwargs)
        # market_info.head()

        for coins in ['bt_', 'eth_']:
            kwargs = {coins+'close_off_high': lambda x: 2*(x[coins+'High'] - x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
                      coins+'volatility': lambda x: (x[coins+'High'] - x[coins+'Low'])/(x[coins+'Open'])}
            market_info = market_info.assign(**kwargs)

        model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_']
                                           for metric in ['Close', 'Volume', 'close_off_high', 'volatility']]]
        model_data = model_data.sort_values(by='Date')
        # print(model_data.head())

        # 切分数据集
        training_set, test_set = model_data[model_data['Date'] <
                                            self.end_date], model_data[model_data['Date'] >= self.end_date]
        training_set = training_set.drop('Date', 1)
        test_set = test_set.drop('Date', 1)

        norm_cols = [coin+metric for coin in ['bt_', 'eth_'] for metric in ['Close', 'Volume']]
        
        training_inputs = []
        print(len(training_set),len(test_set))
        for i in range(len(training_set)-self.window_len):
            temp_set = training_set[i:(i+self.window_len)].copy()
            for col in norm_cols:
                temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
            training_inputs.append(temp_set)
        training_target = (training_set['eth_Close'][self.window_len:].values/training_set['eth_Close'][:-self.window_len].values)-1

        test_inputs = []
        for i in range(len(test_set)-self.window_len):
            temp_set = test_set[i:(i+self.window_len)].copy()
            for col in norm_cols:
                temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
            test_inputs.append(temp_set)
        test_targets = (test_set['eth_Close'][self.window_len:].values/test_set['eth_Close'][:-self.window_len].values)-1

        training_inputs = [np.array(training_input) for training_input in training_inputs]
        training_inputs = np.array(training_inputs)

        test_inputs = [np.array(test_inputs) for test_inputs in test_inputs]
        test_inputs = np.array(test_inputs)

        return training_set,test_set, training_inputs, training_target, test_inputs, test_targets


    def buildModel(self,inputs,output_size,neurons):
        """
        定义网络
        """
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dropout(self.dropout))
        model.add(Dense(output_size,activation='linear'))

        model.compile(loss="mae", optimizer="adam")

        return model
        
    def train(self):
        """
        训练
        """
        training_set,test_set, training_inputs, training_target, test_inputs, test_targets = self.getData()

        eth_model = self.buildModel(training_inputs, 1, 20)
        training_target = (training_set["eth_Close"][self.window_len:].values /
                           training_set['eth_Close'][:-self.window_len].values) - 1

        eth_history = eth_model.fit(training_inputs, training_target,
                                    epochs=self.epochs, batch_size=self.batch_size,
                                    verbose=self.verbose, shuffle=True)

        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(eth_history.epoch, eth_history.history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_ylabel('MAE',fontsize=12)
        ax1.set_xlabel('# Epochs',fontsize=12)
        plt.show()


class Config(object):
    """
    参数配置
    """

    def __init__(self):
        self.loss = "mae"
        self.dropout = 0.25
        self.epochs = 50
        self.batch_size = 1
        self.verbose = 2


if __name__ == '__main__':
    # get_bitcoinlogs()
    opt = Config()
    net = LSTMNet(opt)
    net.train()
 