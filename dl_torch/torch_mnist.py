# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from tensorboardX import SummaryWriter
from skimage import io, transform

torch.manual_seed(1024)

nbatch_size = 128
nclass = 10
learning_rate = 0.001
epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # Conv2d out: (16,28,28)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = torch.nn.ReLU()
        # MaxPool2d out: (16,14,14)
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)     # Conv2d out: (32,14,14)
        self.relu2 = torch.nn.ReLU()
        # MaxPool2d out: (32,7,7)
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(32*7*7, nclass)

    def forward(self, x):
        """
        前向传播
        """
        x = self.relu1(self.conv1(x))
        x = self.maxPool1(x)
        x = self.dropout1(x)

        x = self.relu2(self.conv2(x))
        x = self.maxPool2(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def train():
    """
    训练模型
    """
    train_data, test_data = load_data()
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=nbatch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=nbatch_size, shuffle=True)

    model = CNN()
    model.train(True)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # writer = SummaryWriter('./logs/troch/')
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        for step, (X_train, y_train) in enumerate(train_loader, 0):
            target = y_train.to(device)
            X_train, y_train = X_train.to(device), y_train.to(device)

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
        print('train Loss: {:.4f} \t Acc: {:.4f}'.format(
            train_loss, train_acc), end=' ')

        # writer.add_scalar('data/train_acc', train_acc, global_step=epoch)
        # writer.add_scalar('data/train_loss', train_loss, global_step=epoch)

        # 验证
        model.eval()
        test_loss, test_acc = 0, 0
        for step, (X_test, y_test) in enumerate(test_loader, 0):
            acc_test = y_test.to(device)
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            loss = loss_fn(outputs, y_test)
            y_pred = torch.max(outputs, 1)[1].to(device).data.squeeze()
            test_acc += torch.sum(y_pred == acc_test)
            test_loss += loss.data[0]

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        print('\tval Loss: {:.4f} \t Acc: {:.4f}'.format(test_loss, test_acc))
        print(" ")

    #     writer.add_scalar('data/test_acc', test_acc, global_step=epoch)
    #     writer.add_scalar('data/test_loss', test_loss, global_step=epoch)

    # writer.close()

    # 保存整个神经网络的结构和模型参数
    torch.save(model, "./models/torch/mnist.pkl")
    # 只保存神经网络的模型参数
    torch.save(model.state_dict(), "./models/torch/mnist_params.pth")


def read_one_image(path):
    img = io.imread(path, as_grey=True)
    img = transform.resize(img, (28, 28), mode='reflect')

    return np.asarray(img)


def restore():
    
    cnn = CNN()
    cnn.load_state_dict(torch.load("./models/torch/mnist_params.pth"))
    cnn.to(device)
    print(cnn)
    img = read_one_image("./predict_img/numbers/1.jpg")
    print(img.shape)
    # numpy 转  Tensor
    x = torch.from_numpy(img).float()
    # 扩展 Tensor
    x = x.expand(1,1,img.shape[0],img.shape[1])
    # CUDA
    x = x.to(device)
    print(x.shape)
    outputs = cnn(x)

    _, predicted = torch.max(outputs, 1)
    pred_y = torch.Tensor.argmax(outputs,1).data.cpu().numpy().squeeze()
    print(predicted.data.cpu().numpy().squeeze())
    print(pred_y)


if __name__ == '__main__':
    # main()

    # load_data()

    # train()

    restore()
