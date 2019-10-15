import requests
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
SAVE_PATH = "./test/"
CHAR_SET = number + alphabet + ALPHABET
MAX_CAPTCHA = 4
CHAR_SET_LEN = 62

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def down_images(name):
    store_path = './images/'
    url = 'https://isisn.nsfc.gov.cn/egrantindex/validatecode.jpg?'
    resp = requests.get(url)
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    with open(store_path+ '{}.jpg'.format(name), 'wb') as f:
        f.write(resp.content)


def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = CHAR_SET.index(c)
        vector[i][idx] = 1.0
    return vector


def load_image(name):
    batch_size = 1
    store_path = './images/'
    img = Image.open(store_path + '{}'.format(name))
    # img.show()
    num_img = np.array(img)
    # print(num_img.shape)
    normal_img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
    # normal_img.show()
    num_normal_img = np.array(normal_img)
    # print(num_normal_img.shape)
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA, CHAR_SET_LEN])
    image = tf.reshape(convert2gray(num_normal_img), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    text = '0000'
    for i in range(batch_size):
        batch_x[i, :] = image
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y

def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(CHAR_SET[c])
    return "".join(text)


def predict():
    model = keras.models.load_model('../test_10_15/250.h5')
    if model:
        print('load model success')
    files = os.listdir('./images')
    for i in files:
        datas, labels = load_image(i)
        predict_value = model.predict(datas)
        predict_value = vec2text(np.argmax(predict_value, axis=2)[0])
        print('predict: ', predict_value, '--', i.split('.')[0])
        if predict_value.upper() == i.split('.')[0]:
            print('predict: ', predict_value, 'true: ', i, 'ok----------------')

if __name__ == '__main__':
    # for i in range(10):
    #     down_images(i)
    # load_image()
    predict()
    print('ok ')