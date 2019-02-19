# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:29:08 2019

Using saved CNN model to predicting new images

@author: lizhichuan
"""
import tensorflow as tf

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# getting predictions on val set.
