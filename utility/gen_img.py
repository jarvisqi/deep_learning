# -*- coding: utf-8 -*-
import os
import random
from PIL import Image,ImageFont,ImageDraw


def textImageGenerator(img_w=300, img_h=100):

    while True:
        for i in range(batchSize):
            text = gbk2312()
            im = Image.new("RGB", (img_w, img_h), (255, 255, 255))
            dr = ImageDraw.Draw(im)
            font = ImageFont.truetype("C:\\WINDOWS\\Fonts\\simsun.ttc", 20)
            dr.text((10, 5), text, font=font, fill="#000000")

            path = "./data/img_orc/{}.png".format(text)
            im.save(path)
        yield text, path
    
    return 'done'


def gbk2312():
    len = random.randint(1, 12)
    str = ''
    for i in range(len):
        head = random.randint(0xb0, 0xf7)
        body = random.randint(0xa1, 0xf9)
        val = f'{head:x}{body:x}'
        str += bytes.fromhex(val).decode('gb2312')
    return str


def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'


if __name__ == '__main__':
    # textImageGenerator()

    for x  in textImageGenerator(300,100):
        print(x)

