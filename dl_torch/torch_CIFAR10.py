# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch
import torch.nn.functional as fn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_data():
    """
    加载数据
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

    return train_loader, test_loader


def train(epochs):

    train_loader, test_loader = load_data()

    net = Net()

    # 定义损失函数和优化器
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    all_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (X_train, y_train) in enumerate(train_loader, 0):
            X_train, y_train = Variable(X_train), Variable(y_train)

            optimizer.zero_grad()                   # 清空上一步的残余更新参数值
            outs = net(X_train)                     # 输入训练数据 x, 输出分析值
            loss = loss_fun(outs, y_train)          # 计算两者的误差
            loss.backward()                         # 误差反向传播, 计算参数更新值
            optimizer.step()                        # 将参数更新值施加到 net 的 parameters 上

            running_loss += loss.data[0]
            if i % 2000 == 1999:
                print('[{0}, {1}] loss: {2:.4f}'.format(epoch + 1, i + 1, running_loss / 2000))
                all_losses.append(running_loss / 2000)
                running_loss = 0.0

    # test
    class_correct = [0. for i in range(10)]
    class_total = [0. for i in range(10)]
    for data in test_loader:
        X_test, y_test = data
        outs = net(Variable(X_test))
        _, predicted = torch.max(outs.data, 1)
        acc_sqe = (predicted == y_test).squeeze()
        for i in range(epochs):
            label = labels[i]
            class_correct[label] += acc_sqe[i]
            class_total[label] += 1

    for i in range(classes):
        acc = 100 * class_correct[i] / class_total[i]
        print('Accuracy of {0} : {1:.4f}%'.format(classes[i], acc))

        
    plt.figure()
    plt.plot(all_losses)
    plt.grid(True)
    plt.show()

    print('Finished Training')


class Net(torch.nn.Module):
    """
    构建网络
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        """
        前向传播
        """
        x = self.pool(fn.relu(self.conv1(x)))
        x = self.pool(fn.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = self.fc3(x)

        return x



if __name__ == '__main__':
    
    train(4)

