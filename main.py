#!/usr/bin/env python

import numpy as np
import keras
import tensorflow as tf
from keras.applications import ResNet50
from keras.preprocessing import image as image_utils
from keras.applications.resnet50 import preprocess_input, decode_predictions
import requests
from PIL import Image
from io import BytesIO
import cv2
from flask import Flask, jsonify
app = Flask(__name__)  # create a Flask app

def url_to_image(url, size=224):
    """
    downloads an image from url, converts to numpy array,
    resizes, and returns it
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = image_utils.img_to_array(img)
    img = cv2.resize(img, (224, 224))
    print("expand dim")
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    print("preproc img")
    img = preprocess_input(img)
    return img

def download_and_cache_all():
    # loops through all models and stores weights in ~/.keras/models
    ResNet50(weights='imagenet')
def get_nnet():
    model = ResNet50(weights='imagenet')
    return model

@app.route('/predict/<path:url>', methods=['POST'])
def predict(url):
    top = 1
    size = 224
    img = url_to_image(url)
    pred = model.predict(img)
    res = decode_predictions(pred, top=5)[0]
    print('Predicted:', res)
    return jsonify({'prediction': str(res)})

if __name__ == '__main__':
    download_and_cache_all()
    print('initialize model...')
    model = get_nnet()
    # fix bug as per https://github.com/fchollet/keras/issues/6462
    model._make_predict_function()
    app.run(debug=True)  # this will start a local server
