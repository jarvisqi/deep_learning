# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.autograd import Variable
import torchvision as tv
from tensorboardX import SummaryWriter

torch.manual_seed(1024)


class Config(object):
    """
    超参数
    """

    def __init__(self):
        self.image_size = 96         # 图片尺寸
        self.batch_size = 256
        self.epochs = 200
        self.lr = 2e-4
        self.dataPath = r"F:/ML_Data/Data/faces/"


class NetG(torch.nn.Module):
    """
    定义 Generative 网络
    """

    def __init__(self):

        super(NetG, self).__init__()

        self.convTran1 = torch.nn.ConvTranspose2d(100, 64*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.batNorm1 = torch.nn.BatchNorm2d(64*8)
        self.relu1 = torch.nn.ReLU(True)            # 64*8*4*4

        self.convTran2 = torch.nn.ConvTranspose2d(64*8, 64*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batNorm2 = torch.nn.BatchNorm2d(64*4)
        self.relu2 = torch.nn.ReLU(True)            # 64*4*8*8

        self.convTran3 = torch.nn.ConvTranspose2d(64*4, 64*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batNorm3 = torch.nn.BatchNorm2d(64*2)
        self.relu3 = torch.nn.ReLU(True)            # 64*2*16*16

        self.convTran4 = torch.nn.ConvTranspose2d(64*2, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batNorm4 = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU(True)            # 64*32*32

        self.convTran5 = torch.nn.ConvTranspose2d(64, 3, kernel_size=5, stride=3, padding=1, bias=False)
        self.tanh = torch.nn.Tanh()                 # 3 x 96 x 96

    def forward(self, x):
        """
        前向传播
        """
        x = self.relu1(self.batNorm1(self.convTran1(x)))    # [256, 512, 4, 4]
        x = self.relu2(self.batNorm2(self.convTran2(x)))    # [256, 256, 8, 8]
        x = self.relu3(self.batNorm3(self.convTran3(x)))    # [256, 128, 16, 16]
        x = self.relu4(self.batNorm4(self.convTran4(x)))    # [256, 64, 32, 32]
        x = self.tanh(self.convTran5(x))                    # [256, 3, 96, 96]

        return x



class NetD(torch.nn.Module):
    """
    定义 Discriminative 网络
    """

    def __init__(self):

        super(NetD, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, stride=3, padding=1, bias=False)
        self.lRelu1 = torch.nn.LeakyReLU(0.2, inplace=True)            # 64*32*32

        self.conv2 = torch.nn.Conv2d(64, 64*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batNorm1 = torch.nn.BatchNorm2d(64*2)
        self.lRelu2 = torch.nn.LeakyReLU(0.2, inplace=True)            # (64*2)*16*16

        self.conv3 = torch.nn.Conv2d(64*2, 64*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batNorm2 = torch.nn.BatchNorm2d(64*4)
        self.lRelu3 = torch.nn.LeakyReLU(0.2, inplace=True)            # (64*4)*8*8

        self.conv4 = torch.nn.Conv2d(64*4, 64*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batNorm3 = torch.nn.BatchNorm2d(64*8)
        self.lRelu4 = torch.nn.LeakyReLU(0.2, inplace=True)            # (64*8)*4*4

        self.conv5 = torch.nn.Conv2d(64*8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sig = torch.nn.Sigmoid()                                   # (64*1)*1*1


    def forward(self, x):
        """
        前向传播
        """
        x = self.lRelu1(self.conv1(x))                  # [256, 64, 32, 32]
        x = self.lRelu2(self.batNorm1(self.conv2(x)))   # [256, 128, 16, 16]
        x = self.lRelu3(self.batNorm2(self.conv3(x)))   # [256, 256, 8, 8]
        x = self.lRelu4(self.batNorm3(self.conv4(x)))   # [256, 512, 4, 4]
        x = self.sig(self.conv5(x))                     # [256, 1, 2, 2]
        # x = x.view(x.size(0), -1)                     # [256,1]
        x = x.view(-1)                                  # [256]

        return x



def train():
    opt = Config()
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.dataPath, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             drop_last=True)
    # 定义网络
    netG, netD = NetG(), NetD()
    # 定义优化器和损失函数
    optimizer_g = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    loss_fn = torch.nn.BCELoss()

    # 真图片label为1，假图片label为0
    true_labels = Variable(torch.ones(opt.batch_size))
    fake_labels = Variable(torch.zeros(opt.batch_size))
    # noises为生成网络的输入
    fix_noises = Variable(torch.randn(opt.batch_size, 100, 1, 1))
    noises = Variable(torch.randn(opt.batch_size, 100, 1, 1))

    netG.cuda(), netD.cuda(), loss_fn.cuda()
    true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
    fix_noises, noises = fix_noises.cuda(), noises.cuda()
    
    # 日志可视化
    writer = SummaryWriter(log_dir="./logs/GANs")
    for epoch in range(opt.epochs):
        d_loss, g_loss = 0.0, 0.0
        for step, (img, _) in enumerate(dataloader, 0):
            real_img = Variable(img).cuda()

            # 每1个batch训练一次判别器
            if step % 1 == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                output = netD(real_img)
                error_d_real = loss_fn(output, true_labels)
                error_d_real.backward()

                # 尽可能把假图片判别为错误
                noises.data.copy_(torch.randn(opt.batch_size, 100, 1, 1))
                fake_img = netG(noises).detach()
                output = netD(fake_img)
                error_d_fake = loss_fn(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real
                d_loss += error_d.data[0]

            # 每5个batch训练一次生成器
            if step % 5 == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size, 100, 1, 1))
                fake_img = netG(noises)
                output = netD(fake_img)
                error_g = loss_fn(output, true_labels)
                error_g.backward()
                optimizer_g.step()

                g_loss += error_g.data[0]

            # 每间隔20 batch，展示一次（可视化）
            if step % 20 == 20 - 1:
                fix_fake_imgs = netG(fix_noises)

        # 每10个epoch保存一次
        if epoch % 10 == 0:
            tv.utils.save_image(fix_fake_imgs.data[:64], "{}/{}.png".format("./images/faces", epoch), normalize=True, range=(-1, 1))
            optimizer_g = torch.optim.Adam(netG.parameters(), lr=opt.lr,betas=(0.5,0.999))
            optimizer_d = torch.optim.Adam(netD.parameters(), lr=opt.lr,betas=(0.5,0.999))

        print('Epoch {}/{}'.format(epoch+1, opt.epochs))
        print('-' * 100)
        print('train D_loss: {:.4f} \t G_loss: {:.4f}'.format(d_loss, g_loss))
        writer.add_scalar('data/D_loss', d_loss, global_step=epoch)
        writer.add_scalar('data/G_loss', g_loss, global_step=epoch)
    
    write.close()


if __name__ == '__main__':
    train()
