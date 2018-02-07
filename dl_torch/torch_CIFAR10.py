# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np 
import torch
import torch.nn.functional as fn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_data(batch_size):
    """
    加载数据
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def train(epochs=None, batch_size=32,log_interval=None):

    train_loader, test_loader = load_data(batch_size)

    net = Net()
    net.cuda()
    # 定义损失函数和优化器
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):

        epoch_loss, epoch_acc = 0, 0
        for step, (X_train, y_train) in enumerate(train_loader, 0):
            acc_train = y_train.cuda()
            X_train, y_train = Variable(X_train).cuda(), Variable(y_train).cuda()

            optimizer.zero_grad()                   # 每次迭代清空上一次的梯度
            outputs = net(X_train)                  # 输入训练数据 x, 输出分析值
            loss = loss_fun(outputs, y_train)       # 计算两者的误差
            loss.backward()                         # 误差反向传播, 计算参数更新值
            optimizer.step()                        # 将参数更新值施加到 net 的 parameters 上
            
            predicted = torch.max(outputs, 1)[1].cuda().data.squeeze()
            epoch_acc += torch.sum(predicted == acc_train) / y_train.size(0)
            epoch_loss += loss.data[0]
            if step % log_interval == 0:
                print('Epoch: {} \ttrain loss: {:.4f} \taccuracy: {:.4f}'.format(epoch+1, epoch_loss/log_interval, epoch_acc/log_interval))
                epoch_loss, epoch_acc = 0, 0

        # for step, (X_test, y_test) in enumerate(test_loader, 0):
        #     X_test, y_test = Variable(X_test).cuda(), Variable(y_test).cuda()
        #     outputs = net(X_test)   



    print('Finished Training')


class Net(torch.nn.Module):
    """
    构建网络
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=5)
        # self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(128*5*5, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        """
        前向传播
        """
        x = self.pool(fn.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(fn.relu(self.conv2(x)))
        x = self.dropout(x)
        # x = self.pool(fn.relu(self.conv3(x)))
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)        #  展平多维的卷积图成 (batch_size, 32 * 7 * 7) x.view(-1, 64*5*5)
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)

        return x



if __name__ == '__main__':
    
    train(epochs=64, batch_size=128, log_interval=100)

