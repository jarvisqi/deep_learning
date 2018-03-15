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

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')


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


def train(epochs=None, batch_size=32):
    """
    训练
    """

    train_loader, test_loader = load_data(batch_size)

    net = Net()
    net.train(True)
    net.cuda()
    # 定义损失函数和优化器
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    total_loss=[]
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        for step, (X_train, y_train) in enumerate(train_loader, 0):
            acc_train = y_train.cuda()
            X_train, y_train = Variable(X_train).cuda(), Variable(y_train).cuda()

            optimizer.zero_grad()                   # 每次迭代清空上一次的梯度
            outputs = net(X_train)                  # 输入训练数据 x, 输出分析值
            loss = loss_fun(outputs, y_train)       # 计算两者的误差
            loss.backward()                         # 误差反向传播, 计算参数更新值
            optimizer.step()                        # 将参数更新值施加到 net 的 parameters 上
            
            # 计算损loss和acc
            predicted = torch.max(outputs, 1)[1].cuda().data.squeeze()
            train_acc += torch.sum(predicted == acc_train)
            train_loss += loss.data[0]
            
        train_acc /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 100)
        print('train Loss: {:.4f} \t Acc: {:.4f}'.format(train_loss, train_acc))

        # test
        test_loss, test_acc = test(net, epoch, loss_fun, test_loader)
        total_loss.append((train_loss,test_loss))

    train_loss = [x[0] for x in total_loss]
    test_loss = [x[1] for x in total_loss]
    # 可视化
    visualization(train_loss,test_loss)

    print('Finished Training')


def test(model, epoch, loss_fun, test_loader):
    model.eval()
    test_loss , test_acc = 0, 0
    for step, (X_test, y_test) in enumerate(test_loader, 0):
        acc_test = y_test.cuda()
        X_test, y_test = Variable(X_test).cuda(), Variable(y_test).cuda()
        outputs = model(X_test)
        loss = loss_fun(outputs, y_test)
        predicted = torch.max(outputs, 1)[1].cuda().data.squeeze()
        test_acc += torch.sum(predicted == acc_test)
        test_loss += loss.data[0]

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    print('val Loss: {:.4f} \t Acc: {:.4f}'.format(test_loss, test_acc))
    print(" ")

    return test_loss, test_acc


def visualization(train_loss=None, test_loss=None):
    """
    损失函数可视化
    """
    x = np.arange(1, 17, 1)
    fig, ax = plt.subplots()
    ax.plot(x, train_loss, label='train', color='r')
    ax.plot(x, test_loss,  label='test', color='g')
    plt.legend()
    plt.grid(True)
    plt.show()


class Net(torch.nn.Module):
    """
    构建网络
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
        self.batchNorm1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.batchNorm2 = torch.nn.BatchNorm2d(64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.avgPool = torch.nn.AvgPool2d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(64*6*6, 10)


    def forward(self, x):
        """
        前向传播
        """
        x = self.avgPool(fn.relu(self.batchNorm1(self.conv1(x))))
        x = self.dropout(x)
        x = self.avgPool(fn.relu(self.batchNorm2(self.conv2(x))))
        x = self.dropout(x)
        # x = self.pool(fn.relu(self.batchNorm3(self.conv3(x))))
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)        #  展平多维的卷积图成 (batch_size, 32 * 7 * 7) x.view(-1, 64*5*5)
        x = self.fc1(x)

        return x


if __name__ == '__main__':
    
    train(epochs=16, batch_size=128)


