import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
from skimage import io, transform



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    """自编码
    
    Arguments:
        nn {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """


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
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()                        # 激活函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


def train():
    # 超参数
    EPOCH = 16
    BATCH_SIZE = 128
    LR = 0.001
    N_TEST_IMG = 5          # 到时候显示 5张图片看效果, 如上图一

    train_data = torchvision.datasets.MNIST(root='./data/',
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True,
                                            )
    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    net = AutoEncoder()
    net.to(device)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = nn.MSELoss().to(device)
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.view(-1, 28*28).to(device)
            b_y = x.view(-1, 28*28).to(device)
            b_label = y.to(device)

            encoded, decoded = net(b_x)

            loss = loss_func(decoded, b_y)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

    # 由于pickle序列化是序列了整个对象，而非某个内存地址，因此在反序列化时，也调用了整个序列对象
    # 需要在序列化和反序列化定义相同的函数名称，但内容可以不一样。否则报错如下：
    # AttributeError: Can't get attribute 'sayhi' on <module '__main__' from

    torch.save(net, "./models/torch/auto_encoder.pt")


def read_one_image(path):
    img = io.imread(path, as_grey=True)
    img = transform.resize(img, (28, 28), mode='reflect')

    return np.asarray(img)


def restore():
    model = torch.load("./models/torch/auto_encoder.pt")
    img = read_one_image("./predict_img/numbers/1.jpg")
    x = np.array([img.reshape(-1)])
    print(x.shape)
    x = torch.from_numpy(x).float()
    x = x.to(device)
    output = model(x)
    pred_y = torch.Tensor.argmax(output[0]).data.cpu().numpy().squeeze()
    print(pred_y)


if __name__ == '__main__':
    # train()

    restore()
