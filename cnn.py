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
# 2019.02.21 Add more layers and change image size to increase accurancy of cnn model
# 2019.03.01 Add loop to train model with different parameters

# Questions: how to keep the trained model for later testing purpose?
#            save model as a file and imported to another file.  --solved at 02.19
# Keep all the logs!

# Parameters compare:
# numbers of layers          1,2,3,4,5
# numbers of epoch           1-50
# numbers of batch size      16,32,64
# image size                 32, 64, 128

# Importing the Machine Learning Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os, os.path

#Remove trained model files from last time trainging
model_save_path = "trained_models"
for root, dirs, files in os.walk(model_save_path):
    for file in files:
        os.remove(os.path.join(root, file))
        
# Part 1: Processing the data
#print(os.listdir("dataset/training"))
#Data augmentation (seprate dataset to traing, validation and testing)
datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

# Hyper parameters for model:
# Total:3 * 3 * 5 = 45 models
image_size_Arr = [32, 64, 128]
batch_size_Arr = [16, 32, 64]
layer_MaxNum = 5
epoch_MaxNum = 3

i = 0   # index of image_size
j = 0   # index of batch_size
m = 1   # index of layers

while i < len(image_size_Arr):
    while j < len(batch_size_Arr):
        # Step 1: Data processing
        #seperating data to training and validation
        training_set = datagen.flow_from_directory(r'..\dataset\training',
                                                 target_size = (image_size_Arr[i], image_size_Arr[i]),
                                                 batch_size = batch_size_Arr[j],
                                                 class_mode = 'categorical',
                                                 subset="training")
        
        validation_set = validation_datagen.flow_from_directory(r'..\dataset\validation',
                                            target_size = (image_size_Arr[i], image_size_Arr[i]),
                                            batch_size = batch_size_Arr[j],
                                            class_mode = 'categorical')
        
        
        # Step 2: Initialise the CNN model
        model = Sequential()
        # Add first convolutional layer
        number_filter = 16    # number of features
        model.add(Convolution2D(number_filter, kernel_size=(3,3), input_shape = (image_size_Arr[i], image_size_Arr[i], 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        #Loop to add more convolutional layers
        while m <= layer_MaxNum:
            g = 0
            while g < len(range(1,m)):
                model.add(Convolution2D(number_filter * (m*2), kernel_size=(3,3), activation = 'relu'))
                model.add(MaxPooling2D(pool_size = (2, 2)))
                g += 1
                
            # Step 3 - Flattening
            model.add(Flatten())
            # Step 4 - Full connection
            model.add(Dense(output_dim = 128, activation = 'relu'))
            model.add(Dense(output_dim = 4, activation = 'softmax'))
            
            # Step 5: Compiling the CNN
            model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            
            # Step 6: Train the CNN
            History = model.fit_generator(training_set,
                             samples_per_epoch = 2197,
                             nb_epoch = epoch_MaxNum,
                             validation_data = validation_set,
                             nb_val_samples = 309)
            #Model accuracy
            # summarize history for accuracy
            plt.plot(History.history['acc'])
            plt.plot(History.history['val_acc'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            #plt.show()
            #plt.savefig(model_save_path+"\\IG_"+str(image_size_Arr[i])+"_BH_"+str(batch_size_Arr[j])+"_LR_"+str(m+1)+"_EH_"+str(epoch_MaxNum)+'_plot.png')
            
            # Step 7: save trained model
            # serialize model to JSON
            model_json = model.to_json()
            model_json_name ="IG_"+str(image_size_Arr[i])+"_BH_"+str(batch_size_Arr[j])+"_LR_"+str(m)+"_EH_"+str(epoch_MaxNum)+".json"
            with open(model_save_path+"\\"+model_json_name, "w") as json_file:
                json_file.write(model_json)

            # serialize weights to HDF5
            model_weight_name = "IG_"+str(image_size_Arr[i])+"_BH_"+str(batch_size_Arr[j])+"_LR_"+str(m)+"_EH_"+str(epoch_MaxNum)+".h5"
            model.save_weights(model_save_path+"\\"+model_weight_name)
            msg = "Saved cnn model:""IG_"+str(image_size_Arr[i])+"_BH_"+str(batch_size_Arr[j])+"_LR_"+str(m)+"_EH_"+str(epoch_MaxNum)+" to disk."
            print(msg)
                
            m += 1
        j += 1
    i += 1