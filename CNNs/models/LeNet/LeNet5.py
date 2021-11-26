# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:45:39 2021

@author: yeyit
"""

#import neccesary libraries

import tensorflow as tf # tensorflow 2.0 Google Deep learning release
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten 
from keras.layers import AveragePooling2D
import sys
import os

#add path to parent class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import CNN

class LeNet_CNN(CNN.CNN):
    def __init__(self,dropout=0.25, num_classes=2,data_shape=(32,32,3)):
        CNN.CNN.__init__(self,"LeNet5",'.\CNNs\models\LeNet\checkpoints',dropout=dropout,
                     num_classes=num_classes,data_shape=data_shape)
        

    
    def  createModel(self):

        #create model
        model = Sequential()
        # Layer 1 Conv2D
        model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=self.data_shape, padding="same"))
        # Layer 2 Pooling Layer
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Layer 3 Conv2D
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # Layer 4 Pooling Layer
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(units=120, activation='tanh'))
        model.add(Dense(units=84, activation='tanh'))
        model.add(Dense(units=2, activation='sigmoid'))
        #model.compile(optimizer='sgd',loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
       # Get training and test loss histories
        self.model=model


