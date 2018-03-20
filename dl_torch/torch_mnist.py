# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from tensorboardX import SummaryWriter

torch.manual_seed(1024)  
  
nbatch_size = 128
nclass = 10
learning_rate=0.001
epochs=10

def load_data():
    # 获取数据
    train_data = torchvision.datasets.MNIST(
        root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST(
        root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

    # 打印MNIST数据集的训练集及测试集的尺寸
    print(train_data.train_data.size(), train_data.train_labels.size())
    print(test_data.test_data.size(), test_data.test_labels.size())

    return train_data, test_data


class CNN(torch.nn.Module):
    """
    CNN
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)  # (16,28,28) 
        self.relu1=torch.nn.ReLU()                                       # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2  
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=2)                # (16,14,14)  
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)   # (32,14,14)  
        self.relu2=torch.nn.ReLU()
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=2)               # (32,7,7)  
        self.dropout2 = torch.nn.Dropout(p=0.5)
        
        self.fc1 = torch.nn.Linear(32*7*7, nclass)

    def forward(self, x):
        """
        前向传播
        """
        x = self.maxPool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.maxPool2(self.relu2(self.conv2(x)))
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def train():
    """
    训练模型
    """
    train_data, test_data = load_data()
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=nbatch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=nbatch_size, shuffle=True)

    model = CNN()
    model.train(True)
    model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter('./logs/troch/')
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        for step, (X_train, y_train) in enumerate(train_loader, 0):
            target = y_train.cuda()
            X_train, y_train = Variable(X_train).cuda(), Variable(y_train).cuda()

            optimizer.zero_grad()
            outputs = model(X_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()

            # 返回指定维度dim中每行的最大值，同时返回每个最大值的位置索引。 out[0]:最大值,out[1]:索引
            y_pred = torch.max(outputs, 1)[1].cuda().data.squeeze()
            train_acc += torch.sum(y_pred == target)
            train_loss += loss.data[0]

        train_acc /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 100)
        print('train Loss: {:.4f} \t Acc: {:.4f}'.format(train_loss, train_acc), end=' ')
                
        writer.add_scalar('data/train_acc', train_acc, global_step=epoch)
        writer.add_scalar('data/train_loss', train_loss, global_step=epoch)

        # 验证
        model.eval()
        test_loss , test_acc = 0, 0
        for step, (X_test, y_test) in enumerate(test_loader, 0):
            acc_test = y_test.cuda()
            X_test, y_test = Variable(X_test).cuda(), Variable(y_test).cuda()
            outputs = model(X_test)
            loss = loss_fn(outputs, y_test)
            y_pred = torch.max(outputs, 1)[1].cuda().data.squeeze()
            test_acc += torch.sum(y_pred == acc_test)
            test_loss += loss.data[0]

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        print('\tval Loss: {:.4f} \t Acc: {:.4f}'.format(test_loss, test_acc))
        print(" ")

        writer.add_scalar('data/test_acc', test_acc, global_step=epoch)
        writer.add_scalar('data/test_loss', test_loss, global_step=epoch)

    writer.close()
    
    torch.save(model, "./models/torch/mnist.pkl")    # 保存整个神经网络的结构和模型参数
    torch.save(model.state_dict(), "./models/torch/mnist_params.pkl")   # 只保存神经网络的模型参数


if __name__ == '__main__':
    # main()

    # load_data()

    train()
