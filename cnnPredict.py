# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:29:08 2019

Using saved CNN model to predicting new image

2019.02.19 create file
2019.02.21 change image size for testing since cnn model has been retained.

@author: lizhichuan
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#import cv2                  
#import numpy as np  
#from tqdm import tqdm
#import os                   
#from random import shuffle  
#from zipfile import ZipFile
#from PIL import Image

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Load trained cnn model from disk")

       
# Process testing data
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(r'..\dataset\testing',
                                           target_size = (150, 150),
                                           batch_size = 867,
                                           class_mode = 'categorical')

x_test, y_test = test_set.next()
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)

maximum_index = len(x_test) -1
# index maximum 866 (867 images in testing dataset)
image_index = int(input("Enter a number(Maximum value is " + str(maximum_index) + "):"))

#todo: out of boundary exception handle?
print("You input number "+ str(image_index) + " as index of image.")
print("The Actual flower label is:")
print(y_test[image_index]) 

pred = loaded_model.predict(x_test[image_index].reshape(1, 150, 150, 3))
print("The predicted flower label is:")
print(pred.argmax())

print("Test flower image:")
plt.imshow(x_test[image_index],cmap='Greys')

# Think:
#Since above code doesn't show the label in number not flower names,
# Could consider to set label by ourself (need modify cnn code as well)
#X=[]
#Z=[]
#IMG_SIZE=64
#FLOWER_DAISY_DIR='dataset/testing/daisy'
#FLOWER_SUNFLOWER_DIR='dataset/testing/sunflower'
#FLOWER_TULIP_DIR='dataset/testing/tulip'
#FLOWER_DANDI_DIR='dataset/testing/dandelion'
#FLOWER_ROSE_DIR='dataset/testing/rose'
#
#def assign_label(img,flower_type):
#    return flower_type
#    
#def make_train_data(flower_type,DIR):
#    for img in tqdm(os.listdir(DIR)):
#        label=assign_label(img,flower_type)
#        path = os.path.join(DIR,img)
#        img = cv2.imread(path,cv2.IMREAD_COLOR)
#        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
#        
#        X.append(np.array(img))
#        Z.append(str(label))
# 
#make_train_data('Daisy',FLOWER_DAISY_DIR)
#make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
#make_train_data('Tulip',FLOWER_TULIP_DIR)
#make_train_data('Dandelion',FLOWER_DANDI_DIR)
#make_train_data('Rose',FLOWER_ROSE_DIR)
#print(len(X))  

