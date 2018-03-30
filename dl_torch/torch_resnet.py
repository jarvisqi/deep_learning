import torch
from torch.nn import functional as fn
from tensorboardX import SummaryWriter
import sys
sys.path.append("./")
from utility.visualize import make_dot


class ResidualBlock(torch.nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = torch.nn.Sequential(
            torch.nn.Conv2d(inchannel, outchannel,kernel_size=3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(outchannel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(outchannel, outchannel, kernel_size=3,stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = fn.relu(out)
        return out


class ResNet(torch.nn.Module):

    def __init__(self, num_classes=1000):

        super(ResNet, self).__init__()

        self.pre = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(inchannel=64, outchannel=128, block_num=3)
        self.layer2 = self._make_layer(inchannel=128, outchannel=256, block_num=4, stride=2)
        self.layer3 = self._make_layer(inchannel=256, outchannel=512, block_num=6, stride=2)
        self.layer4 = self._make_layer(inchannel=512, outchannel=512, block_num=3, stride=2)

        self.fc = torch.nn.Linear(512, num_classes)

    
    def _make_layer(self, inchannel, outchannel, block_num, stride=1):

        shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(inchannel, outchannel,kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel,stride=stride, shortcut=shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)         # [1, 64, 56, 56]
        x = self.layer1(x)      # [1, 128, 56, 56]
        x = self.layer2(x)      # [1, 256, 28, 28]
        x = self.layer3(x)      # [1, 512, 14, 14]
        x = self.layer4(x)      # [1, 512, 7, 7]

        x = fn.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def hook(module, inputdata, output):
    '''把这层的输出拷贝到features中'''
    print("钩子输出：", output.data.size())



def main():
    img = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    model = ResNet()
    handle = model.pre[0].register_forward_hook(hook)
    out = model(img)
    print(out)
    g = make_dot(out)
    g.view()


if __name__ == '__main__':
    main()


