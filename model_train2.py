# -*- coding:utf-8 -*-

import numpy as np
from create_images import CreateCaptcha
from PIL import Image
import os
from keras import models, layers
from keras import preprocessing
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


class DataLabel:
    data: list
    label: list


class DataSet:
    datas: list
    labels: list


class ImageLoad(object):
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

    def convert2gray(self, img):
        if len(img.shape) > 2:
            img = np.mean(img, -1)
        return img

    def text2vec(self, text):
        vec = np.zeros((4, len(self.captcha_list)))
        for i in range(len(text)):
            vec[i, self.captcha_list.index(text[i])] = 1
        vec.resize((248,))
        return vec

    def vec2text(self, vec):
        vec.resize((4, 62))
        texts = []
        index = np.where(vec == 1.0)
        # print(index)
        # (array([0, 1, 2, 3]), array([ 1,  2, 10, 11]))
        for j in index[1]:
            texts.append(self.captcha_list[j])
        return ''.join(texts)

    def load_image(self, path):
        # image = Image.open(path)
        # tensor = np.array(image)
        # # tensor = self.convert2gray(tensor)
        # # print(tensor)
        # # print(tensor.shape)
        # # print(tensor.dtype)
        # tensor = tf.image.resize(tensor, [64, 64])
        image = preprocessing.image.load_img(path, target_size=(64, 64))
        tensor = preprocessing.image.img_to_array(image)
        tensor /= 255
        # print(tensor)
        # # tf.Print(tensor)
        # print(tensor.shape)
        name = path.split('/')[-1].split('__')[0]
        data_label = DataLabel()
        data_label.data = tensor
        data_label.label = self.text2vec(name)
        return data_label

    def load_data_set(self):
        files = os.listdir(self.train_path)
        datas = []
        labels = []
        for file in files[:4]:
            path = self.train_path + file
            data_label = self.load_image(path)
            datas.append(data_label.data)
            labels.append(data_label.label)
        data_set = DataSet()
        data_set.datas = np.array(datas)
        data_set.labels = np.array(labels)
        # data_set.datas = np.reshape(data_set.datas, (4, 64*64*3))
        print(data_set.datas.shape)
        print(data_set.labels.shape)
        return data_set

    def gen_load_data_set(self):
        pass


class ModelTrain(object):
    def __init__(self):
        pass

    def model_build(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(126976, activation='relu'))
        model.add(layers.Dense(248, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['acc'])
        model.summary()
        return model

    def model_train(self):
        data_set = ImageLoad().load_data_set()
        datas = data_set.datas
        labels = data_set.labels
        print(len(datas))
        # print(datas)
        print(len(labels))
        print(labels[0].shape)
        print(datas[0].shape)

        model = self.model_build()
        model.fit(datas, labels,
                  epochs=10,
                  batch_size=4)


if __name__ == '__main__':
    # # print(CreateCaptcha().captcha_list)
    # i = ImageLoad()
    # # vec = i.text2vec('02ab')
    # # text = i.vec2text(vec)
    # # print(text)
    # path = i.train_path + '0a00__8920255655840348.jpg'
    # i.load_image(path)
    # # # i.load_data_set()
    ModelTrain().model_train()

