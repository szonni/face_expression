from keras.src.saving import saving_api
import tensorflow as tf
import os
import numpy as np
import cv2

modell = saving_api.load_model(os.path.join('models', 'emote_model.keras'))

img = cv2.imread('data_predicting/happy3.jpeg')

resize_img = tf.image.resize(img, (512, 512))

yhat = modell.predict(np.expand_dims(resize_img / 255, 0))

if yhat < 0.5:
    print("Happy")
else:
    print("Sad")

print(yhat)
