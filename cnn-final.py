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
# 2019.02.19 Add part 5: save trained model to files
# 2019.03.13 trainging the final model with choosen parameters

# Questions: how to keep the trained model for later testing purpose?
#            save model as a file and imported to another file.  --solved at 02.19
# Keep all the logs!

# TIN175  
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
# 2019.02.19 Add part 5: save trained model to files
# 2019.03.13 trainging the final model with choosen parameters


# Part 1 - Building the CNN

# Importing the Machine Learning Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Data processing (training, validation, test dataset)

#hyperparameters setting
batchSize = 32
imageSize = (128, 128)
inputShape = (128, 128, 3)
filterNum = 32

datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

# training dataset
training_set = datagen.flow_from_directory(r'..\dataset\training',
                                                 target_size = imageSize,
                                                 batch_size = batchSize,
                                                 class_mode = 'categorical')

# validation dataset
validation_set = validation_datagen.flow_from_directory(r'..\dataset\validation',
                                            target_size = imageSize,
                                            batch_size = batchSize,
                                            class_mode = 'categorical')

# Test dataset
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(r'..\dataset\test',
                                           target_size = imageSize,
                                           batch_size = batchSize,
                                           class_mode = 'categorical')

x_test, y_test = test_set.next()
x_test = x_test.reshape(x_test.shape[0], 128, 128, 3)

# Part 2: Initialise the CNN model
model = Sequential()

# Adding first Convolutional layer and pooling layer
model.add(Convolution2D(filterNum, kernel_size=(3,3), input_shape = inputShape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer and pooling layer
model.add(Convolution2D(filterNum, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filterNum, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filterNum, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(activation="relu", units=128))

# using dropout to prevent overfittting
model.add(Dropout(0.2))

# output layer
model.add(Dense(activation="softmax", units=4))

model.summary()

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 3 - Fitting the CNN to the images
History = model.fit_generator(training_set,
                             steps_per_epoch = batchSize,
                             epochs = 30,
                             validation_data = validation_set,
                             validation_steps = 309)

# Part 4: Evaluate the model performance

#Model loss
#plt.plot(History.history['loss'])
#plt.plot(History.history['val_loss'])
#plt.title('Model Loss')
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.legend(['train', 'validation'])
#plt.show()

#Model accuracy
plt.plot(History.history['train_acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()



# Part 5: save trained model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model.h5")
print("Save cnn model to disk")


