import os
import sys
import keras
import keras_utils
import greekcaptcha

import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

global model

def ocr_image(imagepath, model):
    image = cv2.imread(imagepath)
    digit_images = greekcaptcha.chop_image(image)

    s = ""
    for digit in digit_images:
        res = model.predict(digit.reshape(1, greekcaptcha.img_rows, greekcaptcha.img_cols, 1))
        s = s + greekcaptcha.LEXICON[res.argmax(1)[0]]
    return s
        

model = keras_utils.load_model("captcha_net/model")

print(ocr_image(sys.argv[1], model))