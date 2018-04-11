# -*- coding: utf-8 -*-
import math
import torch
import torchvision as tv
from tensorboardX import SummaryWriter
from torch.autograd import Variable


class VGG(torch.nn.Module):

    def __init__(self,features,num_classes=1000):
        super(VGG,self).__init__()

        self.conv_block = features
        self.fc_block = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),

            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)            # => [1, 512, 7, 7]
        x = x.view(x.size(0), -1)       # => [1, 25088]
        x = self.fc_block(x)          # =>    [1, 1000]
        return x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.zero_()
            elif isinstance(m, torch.nn.Linear):
                n.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    

def make_layers(config, batch_norm=False):
    layers = []
    in_channels = 3
    for cfg in config:
        if cfg == "M":
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d,torch.nn.BatchNorm2d(cfg), torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = cfg
        
    return torch.nn.Sequential(*layers)


config = {
    "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11():
    model = VGG(make_layers(config["VGG11"]))
    return model


def vgg13():
    model = VGG(make_layers(config["VGG13"]))
    return model


def vgg16():
    model = VGG(make_layers(config["VGG16"]))
    return model


def vgg19():
    model = VGG(make_layers(config["VGG19"]))
    return model


if __name__ == '__main__':
    x = Variable(torch.rand(1, 3, 224, 224))
    model = vgg11()
    # model = vgg13()

    # with SummaryWriter(log_dir="./logs/troch/vgg11", comment='vgg11') as w:
    #     w.add_graph(model, (x, ))
    # print(model)

    y = model(x)
    print(y)
