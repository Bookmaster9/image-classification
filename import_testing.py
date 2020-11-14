#MODEL IMPORT AND TESTING

# run inference on new data

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import model_from_json

import numpy

import os

import matplotlib.pyplot as plt

#load model
loaded_model = tf.keras.models.load_model('/content/drive/My Drive/ai/xray-1')

############## LOAD ENTIRE TEST FOLDER ##############

#Evaluate loaded model on data
image_size = (256, 256)
img = keras.preprocessing.image.load_img(

    "Pediatric Chest X-ray Pneumonia/test/NORMAL/IM-0001-0001.jpeg", target_size=image_size

)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)  # Create batch axis

 

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

predictions = loaded_model.predict(img_array)

score = predictions[0]

################ ADJUST ACCURACY SUMMARY #####################

print(

    "Loaded model: this image is %.2f percent normal and %.2f percent pneumonia."

    % (100 * (1 - score), 100 * score)

)