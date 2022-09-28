"""
Implementing Model: Inception v3 

TODO: Change the directory path before you implement the code. 
This program assumes you have already-trained model2 folder. If you are not using it, please
uncomment and comment some sections as indicated in the corresponding code sections. 

"""

import numpy as np 
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import shutil

from glob import glob 
from shutil import copy
from keras import layers, models
from keras.models import Sequential
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.applications.inception_v3 import InceptionV3 # import inception pretrained model
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

  
def loadData():

    print("Creating train, validate, and test folders from database's 'train' folder")
    
    # There are 10 classes (c0 ~ c9) 
    for i in range(10):
        
        # call the database 
        # TODO: change '/Users/eva.lee/Desktop/data2/imgs/train/' to 
                # your own path where you download your database  
        data_dir = '/Users/eva.lee/Desktop/data2/imgs/train/' + 'c' + str(i) + '/'

        img_train = os.listdir(data_dir) 
        img_labels = os.listdir(data_dir)

        # use train_test_split to split the data
        x, x_test, y, y_test = train_test_split(img_train, img_labels, test_size=0.2, train_size=0.8)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, train_size=0.75)
        
        # make the train, test, validate folders 
        os.makedirs('train/' + 'c' + str(i) + '/',exist_ok=True)
        os.makedirs('test/' + 'c' + str(i) + '/',exist_ok=True)
        os.makedirs('validate/' + 'c' + str(i) + '/',exist_ok=True)
        
        # add the images into three corresponding folders if not yet been added 
        for x in x_train:
            if (not os.path.exists('./train/' + 'c' + str(i) + '/' + x)):
                copy(data_dir + x, './train/' + 'c' + str(i) + '/' + x)

        for x in x_test:
            if (not os.path.exists('./test/' + 'c' + str(i) + '/' + x)):
                copy(data_dir + x, './test/' + 'c' + str(i) + '/' + x)

        for x in x_val:
            if (not os.path.exists('./validate/' + 'c' + str(i) + '/' + x)):
                copy(data_dir + x, './validate/' + 'c' + str(i) + '/' + x)

    print("Data loaded") 

def representation(trainPath):
    # 10 classes labeled from c0 to c9, each containing different set of images 
    classification = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    representation = {'c0': 'safe driving', 'c1': 'texting - right', 'c2': 'talking on the phone - right',
                    'c3': 'texting - left', 'c4': 'talking on the phone - left', 'c5':'operating the radio',
                    'c6': 'drinking', 'c7': 'reaching behind', 'c8': 'hair and makeup',
                    'c9': 'talking to passenger'}

    # plot image 
    plt.figure(figsize=(50, 50)) 
    i = 0 # counter for making the images be aligned to each other side by side  

    # show one image from each class (10 images will be shown in total)
    for classes in classification:
        plt.subplot(1, 10, i + 1) # to plot image data from each class side by side 
        # print(trainPath + '/' + classes + '/*.jpg') # uncomment this if you want to see the name of the image 
        path = glob(trainPath + '/' + classes + '/*.jpg') # grob the image 
        openImage = cv2.imread(path[0], cv2.IMREAD_COLOR) 
        openImage = cv2.cvtColor(openImage, cv2.COLOR_BGR2RGB) 
        plt.imshow(openImage) # show the image 
        plt.title(representation[classes]) # print the title/class of the image 
        plt.axis('off')
        i += 1

def pretrainedModel():
    
    pre_trained_model = InceptionV3(input_shape = (256, 256, 3), # Shape of our input images, 3 as for RGB
                                include_top = False, # Leave out the last fully connected layer
                                weights = 'imagenet') 
    
    #pre_trained_model.summary() # uncomment this if you want to print the structure of this model 
    
    # Flatten the output layer to 1 dimension
    flatten = layers.Flatten()(pre_trained_model.output)
    
    # Add a fully connected layer with 512 hidden units and ReLU activation
    layer = layers.Dense(512, activation='relu')(flatten)
    
    # normalization 
    normal = layers.BatchNormalization()(layer)
    
    # Add a dropout rate of 0.5 
    dropout = layers.Dropout(0.5)(normal)
    
    # Add a fully connected layer with 128 hidden units and ReLU activation
    layer2 = layers.Dense(128, activation='relu')(dropout)
    
    # Add a dropout rate of 0.25 
    dropout2 = layers.Dropout(0.25)(layer2)
    
    # Add a final softmax layer for classification
    x = layers.Dense(10, activation='softmax')(dropout2)

    model = tf.keras.Model(pre_trained_model.input, outputs=x, name='pretrainedModel')

    print("Passed function pretrained model") 

    return model

    
def createPlot(history):
    # Plot the graph in terms of accuracy and loss 
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1) 

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    print("Passed function createPlot")

def evaluator(test_y, pred_y):
  
    # Confusion Matrix
    print("confusion matrix: \n", confusion_matrix(test_y, pred_y))

    # accuracy
    print("accuracy: ", accuracy_score(test_y, pred_y))

    # recall score
    print("recall score: ",recall_score(test_y, pred_y, average='weighted'))

    # precision
    print("precision: ", precision_score(test_y, pred_y, average='weighted'))

    # F1 score 
    print("F1 score: ", f1_score(test_y, pred_y, average='weighted'))




if __name__=="__main__":

    ######################## Load Data ################################################
    if not os.path.exists('./train/'): # create train, test, and validate folders 
        loadData()

    else: # folders exist 
        print("Three folders exist already") 

    # paths for train, validate, test, and predict folders 
    # change to your own path if necessarily 
    trainPath = './train'
    validatePath = './validate'
    testPath = './test'
    predictPath = './testdemo' # currently at small database 

    ######################## Representation of Input ##################################
    representation(trainPath)


    ######################## Data Preprocessing #######################################
    # all images will be rescaled by 1.0 / 255 
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    validate_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    predict_datagen = ImageDataGenerator(rescale=1.0/255)
    test_predict_datagen = ImageDataGenerator(rescale=1.0/255)

    # train using images in train folder
    train_generator = train_datagen.flow_from_directory( 
        trainPath, # the location of train folder
        target_size=(256, 256), # all images will be resized to (256, 256)
        batch_size=20,
        color_mode="rgb",
        class_mode='categorical', # 10 classes
        shuffle=True) # shuffle to increase the accuracy  

    # validate using images in validate folder
    validation_generator = validate_datagen.flow_from_directory(
        validatePath, # the location of validate folder
        target_size=(256, 256),  # all images will be resized to (256, 256)
        batch_size=20,
        color_mode="rgb",
        class_mode='categorical', # 10 classes
        shuffle=True) # shuffle to increase the accuracy 

    test_generator = test_datagen.flow_from_directory(
        testPath, # the location of validate folder
        target_size=(256, 256),  # all images will be resized to (256, 256)
        batch_size=20,
        color_mode="rgb",
        class_mode='categorical', # 10 classes
        shuffle=True) # shuffle to increase the accuracy  

    # small dataset is used for prediction for runnability  
    predict_generator = predict_datagen.flow_from_directory(
        predictPath, # the location of prediction folder
        target_size=(256, 256),  # all images will be resized to (256, 256)
        batch_size=20,
        color_mode="rgb",
        class_mode='categorical', # 10 classes
        shuffle=False) # no point to shuffle 

    test_predict_generator = test_predict_datagen.flow_from_directory(
        predictPath, # the location of prediction folder
        target_size=(256, 256),  # all images will be resized to (256, 256)
        batch_size=20,
        color_mode="rgb",
        class_mode='categorical', # 10 classes
        shuffle=False) # no point to shuffle  

    print("Passed Data Preprocessing")

    ######################## Model Training #########################################
    model = pretrainedModel()

    model.compile(
        optimizer="rmsprop", # rmsprop optimizer is used 
        loss='categorical_crossentropy', # 10 classes 
        metrics=['accuracy', 'AUC'])

    batch_size = 512
    steps_per_epoch = 13479 // batch_size
    validation_steps = 4487 //batch_size

    '''
    # uncommnet this if you want to run the model 
    history = model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs=100, 
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps) 
    '''

    # Comment the line below if you want to run the model, instead of saved model 
    model = models.load_model("./model2")
        
    print("Passed Model Training")

    ######################## Evaluate Model #########################################
    '''
    # uncommnet this if you want to run the model 
    model.evaluate(test_generator, batch_size=2, verbose=1)
    ''' 

    ######################## Plot Results ###########################################
    '''
    # uncommnet this if you want to run the model 
    createPlot(history)
    '''

    ######################## Prediction using small dataset #########################
    # predict using the model 
    pred = model.predict(predict_generator, batch_size=10) 
    pred_y = pred.argmax(axis=1)
    y_test = test_predict_generator.labels # get the true lables 
    print("Passed Predict")

    ######################## Results and Evaluation #################################
    evaluator(y_test, pred_y) 
    
    