import os

import torch
import torchvision as tv
from PIL import Image
from torch.autograd import Variable


class Config(object):
    """超参数以及其他配置
    Arguments:
        object {[type]} -- [description]
    """

    data_root = "./data/train/"
    image_size = 224
    batch_size = 64,
    lr = 0.001
    steps = 16


class CustDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """自定义数据集读取
        
        Arguments:
            root {[string]} -- 数据集文件夹路径
        
        Keyword Arguments:
            transforms {[type]} -- torchvision.transforms (default: {None})
            train {bool} -- [description] (default: {True})
            test {bool} -- [description] (default: {False})
        """

        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num):]

        self.transforms = transforms

    def __getitem__(self, index):
        """返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        Arguments:
            index {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        img_path = self.imgs[index]
        if self.test:
             label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
             label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)


def load_data(opt: Config):
    """读取数据
    
    Arguments:
        opt {Config} -- [description]
    """

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CustDataset(opt.data_root, transforms=transforms, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2,
                                               shuffle=True,
                                               num_workers=4)

    
    return train_loader
    
    # for step, (data, label) in enumerate(train_loader):
    #     print(label)
    #     tv.utils.save_image(data, "{}/{}_{}_{}.png".format("./images/Cat",step,label[0], label[1]), normalize=True, range=(-1, 1))
    #     if step > 10:
    #         break



if __name__ == '__main__':
    main()
