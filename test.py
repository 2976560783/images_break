from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import os
path = './captcha_fonts/'
files = os.listdir(path)

path_tos = [os.path.join(path, file) for file in files]
for i in range(3):
    img = ImageCaptcha(fonts=path_tos)
    captcha = img.generate('syau')
    img = Image.open(captcha)
    img.show()
# capt = np.array(img)
# img = Image.fromarray(capt)
# img.show()



