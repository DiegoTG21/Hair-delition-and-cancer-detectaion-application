# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:45:39 2021

@author: diego
"""

#import neccesary libraries

import tensorflow as tf # tensorflow 2.0 Google Deep learning release

from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D,Dropout , Dense ,BatchNormalization

from datetime import *
import sys
import os

#add path to parent class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import CNN

class EfficientNet_CNN (CNN.CNN):
    def __init__(self,dropout=0.5, num_classes=2,data_shape=(224,224,3)):
        CNN.CNN.__init__(self,"EfficientNetB0",'.\CNNs\models\EfficientNet\checkpoints',dropout=dropout,
                     num_classes=num_classes,data_shape=data_shape)
        
    
    def createModel(self):
        #create efficientNetB0 layers
        efficientNet=tf.keras.applications.EfficientNetB0(
            input_shape=self.data_shape,
            classes=self.num_classes,
            include_top=False,
        )
        model = Sequential()
        model.add(efficientNet)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(units = 128, activation = 'relu'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(Dense(2, activation='sigmoid'))

        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

        self.model=model
