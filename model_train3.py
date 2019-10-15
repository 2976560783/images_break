import enable_gpu_grow
import tensorflow as tf
from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

import time
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
from_path = './test_10_15/250.h5'
SAVE_PATH = "./test_10_15/"

CHAR_SET = number + alphabet + ALPHABET
# CHAR_SET = number
CHAR_SET_LEN = len(CHAR_SET)
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160


if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


def random_captcha_text(char_set=None, captcha_size=4):
    if char_set is None:
        char_set = number + alphabet + ALPHABET

    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(width=160, height=60, char_set=CHAR_SET):
    path = './captcha_fonts/'
    files = os.listdir(path)
    path_fonts = [os.path.join(path, file) for file in files]
    image = ImageCaptcha(width=width, height=height, fonts=path_fonts)

    captcha_text = random_captcha_text(char_set)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


text, image = gen_captcha_text_and_image(char_set=CHAR_SET)
MAX_CAPTCHA = len(text)
print('CHAR_SET_LEN=', CHAR_SET_LEN, ' MAX_CAPTCHA=', MAX_CAPTCHA)


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = CHAR_SET.index(c)
        vector[i][idx] = 1.0
    return vector


def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(CHAR_SET[c])
    return "".join(text)


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA, CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(char_set=CHAR_SET)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        img = Image.fromarray(image)
        # img.show()
        image = tf.reshape(convert2gray(image), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        # print(image.shape)
        # print(image)
        # print(text2vec(text).shape)
        batch_x[i, :] = image
        batch_y[i, :] = text2vec(text)
        # time.sleep(1)

    return batch_x, batch_y


def model_build():
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(keras.layers.Conv2D(64, (5, 5)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(keras.layers.Conv2D(128, (5, 5)))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN))
    model.add(keras.layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]))
    model.add(keras.layers.Softmax())

    return model


def train():
    try:
        # files = os.listdir(SAVE_PATH)
        # files.sort()
        # print(files)
        # file = files.pop()
        model = keras.models.load_model(from_path)
        if model:
            print('load model success')
            print(from_path)
    except:
        model = model_build()

    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')

    for times in range(500000):
        batch_x, batch_y = get_next_batch(512)
        train_datas = batch_x[:400]
        train_labels = batch_y[:400]
        val_datas = batch_x[400:]
        val_labels = batch_y[400:]
        print('times=', times, ' batch_x.shape=', batch_x.shape, ' batch_y.shape=', batch_y.shape)
        model.fit(train_datas, train_labels, epochs=4,
                  validation_data=(val_datas, val_labels))
        # 下面一段程序表示是否在训练的时候进行识别
        # correct_count = 0
        # compares = []
        # for i in np.argmax(model.predict(batch_x), axis=2):
        #     compares.append(vec2text(i))
        # for i in np.argmax(batch_y, axis=2):
        #     compares.append(vec2text(i))
        # for i in range(len(compares)//2):
        #     predict_value = compares[i]
        #     true_value = compares[i + 512]
        #     if not i % 100:
        #         print('predict is:', predict_value, '   true is:', true_value)
        #     if predict_value.upper() == true_value.upper():
        #         correct_count += 1
        #         print('predict is:', predict_value, '   true is:', true_value)
        #         print('okkkkkkkkkkkk------------------------------------------------------------------')
        # print('准确度为', correct_count/512)

        if 0 == times % 10:
            print("save model at times=", times)
            model.save(SAVE_PATH + '{}.h5'.format(times))


def predict():
    files = os.listdir(SAVE_PATH)
    files.sort()
    print(files)
    file = files.pop()
    model = keras.models.load_model(SAVE_PATH + '/' + file)
    if model:
        print('load model success')
        print(file)
    success = 0
    count = 1000
    for _ in range(count):
        data_x, data_y = get_next_batch(1)
        prediction_value = model.predict(data_x)
        data_y = vec2text(np.argmax(data_y, axis=2)[0])
        prediction_value = vec2text(np.argmax(prediction_value, axis=2)[0])

        if data_y.upper() == prediction_value.upper():
            print("y预测=", prediction_value, "y实际=", data_y, "预测成功。")
            success += 1
        else:
            print("y预测=", prediction_value, "y实际=", data_y, "预测失败。")

    print("预测", count, "次", "成功率 =", success / count)


if __name__ == "__main__":
    train()
    # predict()
