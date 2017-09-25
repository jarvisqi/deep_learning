import os
import shutil


def main():
    count = 0
    total = 0
    path = './data/faces/'
    for f in os.listdir(path):
        name = f.split(' ')[0]
        os.makedirs(path + name)
        oldPath = path + f
        newPath = path + name + '/' + f
        shutil.copyfile(oldPath, newPath)


if __name__ == '__main__':
    main()
