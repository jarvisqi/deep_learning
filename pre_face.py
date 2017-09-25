

import os
import shutil


def main():
    count=0
    total=0
    path = './dl/text/lfw/'
    for file in os.listdir(path):
        old_name = path + file
        list_file = os.listdir(old_name)
        if len(list_file) > 4:
            count +=1
            total +=len(list_file)
            # new_name = './dl/text/face/' + file
            # shutil.copytree(old_name, new_name)
    print(count,total)

if __name__ == '__main__':

    main()
