# -*- coding: utf-8 -*-
import torch
import torchvision as tv
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import functional as fn
import sys

sys.path.append("./utility/")
from pytorchviz import make_dot

class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()

        self.left = torch.nn.Sequential(
            torch.nn.Conv2d(inchannel, outchannel,kernel_size=3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(outchannel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = fn.relu(out)
        return out


class Bottleneck(torch.nn.Module):
    expansion = 4    
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()

        self.left = torch.nn.Sequential(
            torch.nn.Conv2d(inchannel, outchannel,kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(outchannel),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(outchannel, outchannel, kernel_size=3,stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(outchannel),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(outchannel, outchannel*4,kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(outchannel*4)
        )
        self.right = shortcut
        
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = fn.relu(out, inplace=True)
        return out



class ResNet(torch.nn.Module):
    """
    ResNet
    """

    def __init__(self, block, layers, num_classes=1000):
        self.inchannel = 64
        super(ResNet, self).__init__()

        self.pre = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(7)
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, outchannel, block_num, stride=1):
        shortcut = None
        if stride != 1 or self.inchannel != outchannel * block.expansion:
            out_dim = outchannel * block.expansion
            shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(self.inchannel, out_dim, kernel_size=1,stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_dim)
            )

        layers = []
        layers.append(block(self.inchannel, outchannel,stride=stride, shortcut=shortcut))
        self.inchannel = outchannel * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.inchannel, outchannel))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)         # [1, 64, 56, 56]

        x = self.layer1(x)      # [1, 128, 56, 56]
        x = self.layer2(x)      # [1, 256, 28, 28]
        x = self.layer3(x)      # [1, 512, 14, 14]
        x = self.layer4(x)      # [1, 512, 7, 7]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class Config(object):
    """
    超参数
    """
    image_size = 100         # 图片尺寸
    batch_size = 128
    epochs = 16
    lr = 1e-2
    dataPath = r"F:/ML_Data/Data/faces/"


def load_data(opt: Config):
    
    train_transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.RandomCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.RandomCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 获取数据
    train_data = tv.datasets.CIFAR10(root="./data", train=True, transform=train_transforms, download=False)
    test_data = tv.datasets.CIFAR10(root="./data", train=False, transform=test_transforms, download=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=100, shuffle=False, num_workers=2)
    # 打印CIFAR100数据集的训练集及测试集的尺寸
    # print(train_data.train_data.size(), train_data.train_labels.size())
 
    return train_loader, test_loader


def train():
    """
    训练模型
    """
    opt = Config()
    train_loader, test_loader = load_data(opt)

    resNet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    # resNet34 = ResNet(BasicBlock,block_num=[3,4,6,3])
    # resNet50 = ResNet(Bottleneck,block_num=[3,4,6,3])
    # resNet101 = ResNet(Bottleneck,block_num=[3,4,23,3])
    # resNet152 = ResNet(Bottleneck,block_num=[3,8,36,3])
    resNet18.train(True)
    resNet18.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resNet18.parameters(), lr=opt.lr)

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    for epoch in range(opt.epochs):
        train_loss, train_acc = 0.0, 0.0
        for step, (X_train, y_train) in enumerate(train_loader, 0):
            target = y_train.cuda()
            X_train, y_train = Variable(X_train).cuda(), Variable(y_train).cuda()

            optimizer.zero_grad()
            outputs = resNet18(X_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()
            # 返回指定维度dim中每行的最大值，同时返回每个最大值的位置索引。 out[0]:最大值,out[1]:索引
            y_pred = torch.max(outputs, 1)[1].cuda().data.squeeze()
            train_acc += torch.sum(y_pred == target)
            train_loss += loss.data[0]
            
        train_acc /= train_size
        train_loss /= train_size
        print('Epoch {}/{}'.format(epoch+1, opt.epochs),
              "\t", 'Train Loss: {:.4f} \t Acc: {:.4f}'.format(train_loss, train_acc),end=" ")

        # 验证
        resNet18.eval()
        test_loss , test_acc = 0, 0
        for step, (X_test, y_test) in enumerate(test_loader, 0):
            acc_test = y_test.cuda()
            X_test, y_test = Variable(X_test).cuda(), Variable(y_test).cuda()
            outputs = resNet18(X_test)
            loss = loss_fn(outputs, y_test)
            y_pred = torch.max(outputs, 1)[1].cuda().data.squeeze()
            test_acc += torch.sum(y_pred == acc_test)
            test_loss += loss.data[0]

        test_loss /= test_size
        test_acc /= test_size
        print('\tVal Loss: {:.4f} \t Acc: {:.4f}'.format(test_loss, test_acc))
        print('-' * 100)



if __name__ == '__main__':
    # main()

    # train()
    x = Variable(torch.rand(1, 3, 224, 224))
    # resNet18 = ResNet(BasicBlock,[2, 2, 2, 2])
    # resNet34 = ResNet(BasicBlock,[3,4,6,3])
    resNet50 = ResNet(Bottleneck, [3, 4, 6, 3])
    # resNet101 = ResNet(Bottleneck, [3, 4, 23, 3])

    with SummaryWriter(log_dir="./logs/troch",comment='resNet50') as w:
        w.add_graph(resNet50, (x, ))
    # y = resNet50(x)
    # # model = tv.models.resnet50()
    # # y = model(x)
    # print(y)
    # dot = make_dot(y, name="./images/resNet50",params=dict(list(resNet50.named_parameters()) + [('x', x)]))
    # # dot = make_dot(y, name="./images/resNet50",params=dict(list(model.named_parameters()) + [('x', x)]))
    # dot.format = "pdf"
    # dot.render()
