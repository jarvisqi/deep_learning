import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)                    # 压缩成3个特征, 进行 3D 图像可视化
        )

        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            nn.Sigmoid()                        # 激活函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


def train():
    # 超参数
    EPOCH = 10
    BATCH_SIZE = 64
    LR = 0.005
    N_TEST_IMG = 5          # 到时候显示 5张图片看效果, 如上图一

    train_data = torchvision.datasets.MNIST(root='./data/',
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True,
                                            )
    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    net = AutoEncoder()
    net.cuda()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28*28)).cuda()
            b_y = Variable(x.view(-1, 28*28)).cuda()
            b_label = Variable(y).cuda()

            encoded, decoded = net(b_x)

            loss = loss_func(decoded, b_y)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])


if __name__ == '__main__':
    train()
