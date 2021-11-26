# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:22:25 2021

@author: diego
"""
#import neccesary libraries

import matplotlib.pyplot as plt
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint,ReduceLROnPlateau
from keras.utils import plot_model

from datetime import *
import cv2 as cv
import numpy as np
import os

#import my funcs

import functions.graphsGenerator as graphsGenerator
#all the cnns will inherite from this class 
class CNN:
    #take parameters here so every model created will be built the same way
    def __init__(self,CNNname, checkpoint_filepath,dropout, num_classes,data_shape):
        self.CNNname=CNNname
        self.dropout=dropout
        self.num_classes=num_classes
        self.data_shape=data_shape
        self.checkpoint_filepath=checkpoint_filepath
        #adjust learning rate when no progress is made after a few epochs
        self.lrr= ReduceLROnPlateau( monitor='val_loss',
                                    factor=0.5,
                                    patience=4,
                                    min_lr=1e-6)
        self.csv_log = CSVLogger("./data/results.csv",
                                 append=True)
        #if it doesnt improve after a few epochs by 0.001 or more stop training
        self.early_stopping = EarlyStopping(patience=6
                                            ,min_delta=0.001)
        #save the best performing model
        self.checkpoint = ModelCheckpoint( filepath=checkpoint_filepath,
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
        self.optimizer = Adam(lr=0.0001)
        self.batchSize=16
    
    #load any previously trained models
    def loadModel(self,filepath):
        try:
            self.model=load_model(filepath)
            print(self.CNNname, " model loaded successfully!")
        except Exception as e:
            print("Could not load the model. Check filepath")
            print(e)
    #load a checkpoint (best accuracy epoch)
    def loadCheckpoint(self):
        try:
            self.model=load_model(self.checkpoint_filepath)
            print(self.CNNname, " model loaded successfully!")
        except Exception as e:
            print("Could not load the model. Check filepath")
            print(e)
            
    #        
    def modelScore(self,testImgs, testingLabels):# evaluate the model
        score = self.model.evaluate(testImgs, testingLabels, verbose=1)
        # print performance
        print()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score
    #show a ROC cuurve and a confusion mat 
    def showPerformance(self,x_test,y_test,classes=[1,0]):
        #reshape pictures to the ideal size for the classifier
        if x_test.shape[1:] != self.data_shape:
            _,x_test=CNN.reshapeImgs(self,x_test,x_test)
            print("Pictures were reshaped")
            
        yScores = self.model.predict(x_test)
        graphsGenerator.ROC(yScores,y_test)
        
        predictions = self.model.predict_classes(x_test)
        graphsGenerator.confusionMat(predictions,x_test,y_test,classes)
        

        CNN.modelScore(self,x_test, y_test)
        
    #reshape pictures 
    def reshapeImgs(self,x_train,x_test):
        new_x_train=[]
        new_x_test=[]
        for img in x_train:
            new_img= cv.resize(img, (self.data_shape[0],self.data_shape[1]))
            new_x_train.append(new_img)
            
        for img in x_test:
            new_img= cv.resize(img, (self.data_shape[0],self.data_shape[1]))
            new_x_test.append(new_img) 
            
        new_x_train=np.array(new_x_train)
        new_x_test=np.array(new_x_test)

        return new_x_train,new_x_test
    #train model, store history and show results once finished
    def trainModel(self,x_train,x_test,y_train,y_test,classes=[1,0],epochNum=15):
        if x_train.shape[1:] != self.data_shape:
                x_train,x_test=CNN.reshapeImgs(self,x_train,x_test)
                print("Pictures were reshaped")
        print("Training data shape: " + str(x_train.shape))
        #train the model
        self.history= self.model.fit(x_train,y_train, validation_data=(x_test, y_test),batch_size=self.batchSize,
                           epochs=epochNum, callbacks=[self.early_stopping, self.lrr,self.csv_log,self.checkpoint])

        #show performance
        CNN.showAllGraphs(self, x_test, y_test)
       
    #get a summary of of the model and a diagram showing the structure    
    def showInfo(self):
        try:
            self.model.summary()
            plot_model(self.model, to_file= self.CNNname + '.png', show_shapes=True, show_layer_names=True)
            print("Diagram has been saved as ' " + self.CNNname + ".png'.")
        except Exception as e:
           print("Could not show any information. Model might not exist")
           print(e) 
           
    #save model showing the accuracy and data       
    def saveModel(self):
        try:
            filename = (self.CNNname+"-" + datetime.now().strftime("%d-%m-%Y")+" Acc-%0.3f" %
                        self.val_accuracy[len(self.val_accuracy)-1])
            trainedDir="./models/" + self.CNNname +"/trainedModels/"
            if os.path.exists(trainedDir):#save in existing file, dont create a new one
                self.model.save(trainedDir + filename)

            else:
                self.model.save("./CNNs/models/" + self.CNNname +"/trainedModels/" + filename)

        except Exception as e:
            print("Could not save model")
            print(e)
    #display the training loss, accurary, con. mat. and ROC
    def showAllGraphs(self, x_test, y_test):
        self.accuracy = self.history.history['binary_accuracy']
        self.val_accuracy = self.history.history['val_binary_accuracy']
        
        # Visualize accuracy history

        epoch_count = range(1, len(self.val_accuracy) + 1)
        plt.plot(epoch_count, self.accuracy, 'r--', label='Training accuracy')
        plt.plot(epoch_count, self.val_accuracy, 'b-', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Binary Accuracy')
        plt.legend()
        plt.figure()
        # Get training and test loss histories
        self.training_loss = self.history.history['loss']
        self.test_loss = self.history.history['val_loss']
        
        # Create count of the number of epochs
        epoch_count = range(1, len(self.training_loss) + 1)
        
        # Visualize loss history
        plt.plot(epoch_count, self.training_loss, 'r--')
        plt.plot(epoch_count, self.test_loss, 'b-')
        plt.title("")
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and validation loss')

        plt.show()
        #show performance
        CNN.showPerformance(self, x_test, y_test)
        
    def predictImgs(self,imgs,note=''):
        if note=='':
            note=self.CNNname
        if imgs.shape[1:] != self.data_shape:
            imgs,_=CNN.reshapeImgs(self,imgs,imgs)
            print("Pictures were reshaped")    
        predictions=self.model.predict(imgs)
        graphsGenerator.showPrediction(imgs,predictions, note=note)
        return predictions