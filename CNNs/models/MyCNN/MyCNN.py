# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:22:25 2021

@author: diego 
"""
#import neccesary libraries

import tensorflow as tf # tensorflow 2.0 Google Deep learning release
from tensorflow.keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten ,BatchNormalization
from keras.layers import MaxPooling2D, Dropout
import sys
import os

##############################CODE WAS DELETED AS I AM STILL WORKING ON RESEARCH##################################
#add path to parent class
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import CNN
class MyCNN(CNN.CNN):
        #take parameters here so every model created will be built the same way
        #size of the kernels can be adjusted
    def __init__(self,firstLayerKernelSize=3,secondLayerKernelSize=3, thirdLayerKernelSize=3,forthLayerKernelSize=3,dropout=0.25, num_classes=2,data_shape=(128,128,3)):
        CNN.CNN.__init__(self,"myCNN",'.\CNNs\models\MyCNN\checkpoints',dropout=dropout,
                     num_classes=num_classes,data_shape=data_shape)
        self.firstLayerKernelSize=firstLayerKernelSize
        self.secondLayerKernelSize=secondLayerKernelSize
        self.thirdLayerKernelSize=thirdLayerKernelSize
        self.forthLayerKernelSize=forthLayerKernelSize

    def createModel(self):
        
        #generate model
        model = Sequential()#add model layers
        #create layer & dont change size
        model.add(Conv2D(8, (self.firstLayerKernelSize,self.firstLayerKernelSize), input_shape=self.data_shape, padding='same',activation=LeakyReLU(alpha=0.3)))
        
            # add 2D pooling layer reduces size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #normalise inputs
        model.add(BatchNormalization())
        # add second convolutional layer with 20 filters
        model.add(Conv2D(16, (self.secondLayerKernelSize, self.secondLayerKernelSize), activation=LeakyReLU(alpha=0.3)))
            
        # add 2D pooling layer reduces size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # apply dropout with rate 0.25
        model.add(Dropout(self.dropout))
        #add another layer
        model.add(Conv2D(32, (self.thirdLayerKernelSize, self.thirdLayerKernelSize), padding='same',activation=LeakyReLU(alpha=0.3)))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        #normalise inputs
        model.add(BatchNormalization())
        # flatten data
        model.add(Flatten())
            
        # add a denselayer with 256 neurons
        model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
        
        # apply dropout with rate 0.25
        model.add(Dropout(self.dropout))
        #normalise inputs
        model.add(BatchNormalization())
        
        # softmax layer
        model.add(Dense(self.num_classes, activation='sigmoid'))
        #compile model using accuracy to measure model performance
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
        self.model=model
        return model


    # def trainModel(self,x_train,x_test,y_train,y_test,classes=[1,0],epochNum=15):
    #     #train the model
    #     self.history= self.model.fit(x_train,y_train, validation_data=(x_test, y_test),
    #                        epochs=epochNum, callbacks=[self.early_stopping, self.lrr,self.csv_log,self.checkpoint])
    #     # Get training and test loss histories
    #     training_loss = self.history.history['loss']
    #     test_loss = self.history.history['val_loss']
        
    #     # Create count of the number of epochs
    #     epoch_count = range(1, len(training_loss) + 1)
        
    #     # Visualize loss history
    #     plt.plot(epoch_count, training_loss, 'r--')
    #     plt.plot(epoch_count, test_loss, 'b-')
    #     plt.title("")
    #     plt.legend(['Training Loss', 'Test Loss'])
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.show()
    #     #show performance
    #     CNN.showPerformance(self, x_train, x_test, y_train, y_test)
        