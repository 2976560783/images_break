# coding:utf-8

import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
import os
from concurrent.futures import ProcessPoolExecutor


class CreateCaptcha(object):
    def __init__(self):
        self.NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                         'u',
                         'v', 'w', 'x', 'y', 'z']
        self.UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U',
                        'V', 'W', 'X', 'Y', 'Z']

        self.captcha_list = self.NUMBER + self.LOW_CASE + self.UP_CASE
        self.captcha_len = 4  # 验证码长度
        self.captcha_width = 160
        self.captcha_height = 60
        self.train_path = './images/train/'
        self.test_path = './images/test/'

    def create_image(self, captcha_text, path):
        """
        生成随机验证码
        :param width: 验证码图片宽度
        :param height: 验证码图片高度
        :param save: 是否保存（None）
        :return: 验证码字符串，验证码图像np数组
        """
        image = ImageCaptcha(width=self.captcha_width, height=self.captcha_height)
        if not os.path.exists(path):
            os.mkdir(path)
        random_name = str(random.random()).split('.')[-1]
        image.write(captcha_text, path + captcha_text + '__{}.jpg'.format(random_name))
        print(captcha_text)

    def run(self):
        # 验证码文本
        captcha_texts = []
        for i in self.captcha_list:
            for j in self.captcha_list:
                for k in self.captcha_list:
                    for x in self.captcha_list:
                        captcha_texts.append(''.join([i, j, k, x]))
        with ProcessPoolExecutor(max_workers=500) as worker:
            for captcha_text in captcha_texts:
                worker.submit(self.create_image(captcha_text, self.train_path))


if __name__ == '__main__':
    create = CreateCaptcha()
    create.run()

