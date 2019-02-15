# TIN175-AI-Project  
# Group Pong
# We use convolotional neural network(CNN) for flowers recognition.
# dataset used in this program: https://www.kaggle.com/alxmamaev/flowers-recognition/version/2

# 2019.02.14 first version   accuracy: 0.70
# 2019.XX.XX revised version: ways to imporove accuracy
#            larger piexl from 64 --> ?;
#            pre-processing (clean data);
#            cross-validation
#            learning rate?
# 2019.02.15 Adding plot to show loss and accuracy

# Questions: how to keep the trained model for later testing purpose?
#            save model as a file and imported to another file.
# Keep all the logs!

# Part 1 - Building the CNN

# Importing the Machine Learning Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Initialise the CNN model
model = Sequential()

# Step 1 - Add Convolutional layer
model.add(Convolution2D(16, kernel_size=(3,3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Add Pooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer and pooling layer
model.add(Convolution2D(16, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 5, activation = 'softmax'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

#Data augmentation (seprate dataset to traing, validation and testing)
datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = datagen.flow_from_directory(r'dataset\training',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

validation_set = datagen.flow_from_directory(r'dataset\validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

testing_set = datagen.flow_from_directory(r'dataset\testing',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# Part 2 - Fitting the CNN to the images
History = model.fit_generator(training_set,
                             samples_per_epoch = 2589,
                             nb_epoch = 10,
                             validation_data = validation_set,
                             nb_val_samples = 868)

# Evaluate the model performance

#Model loss
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

#Model accuracy
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# Testing: predict new flower image
# code to be written here...


