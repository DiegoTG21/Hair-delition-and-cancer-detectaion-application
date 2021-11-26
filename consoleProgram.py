# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:12:57 2021

@author: yeyit
"""
#import neccesary libraries
import numpy as np
import tensorflow as tf # tensorflow 2.0 Google Deep learning release
import cv2 as cv
import os
from sklearn.model_selection import train_test_split#randomises
import pandas as pd
#import my funcs

from functions import hairReplacement
from functions import testing
from functions import customedProcessing
from functions import graphsGenerator

#import cnns
import CNNs.models.LeNet.LeNet5 as LeNet5
import CNNs.models.MyCNN.MyCNN as MyCNN
import CNNs.models.EfficientNet.efficientNet as efficientNet
#global variables
num_classes=2

dataDir=r"C:/Users/yeyit/Documents/FinalProject/ISIC_2019_Training_Input" 
groundTruthDir=r'C:/Users/yeyit/Documents/FinalProject/ISIC_2019_Training_GroundTruth.csv'
labels=[]


#options
menu_options = {
    1: 'Show hair removal steps',
    2: 'Show processing steps',
    3: 'See my CNN model summary',
    4: 'Use previously trained models for testing',
    5: 'Small training demo',  
    6: 'Predict class',
    7: 'Exit',
}

processing_menu_options = {
    1: 'Raw imgs.',
    2: 'Even ratio raw imgs.',
    3: 'Even ratio borderless imgs.',
    4: 'Even ratio hairless imgs.',  
    5: 'Even ratio borderless & hairless imgs.',
    6: 'Even ratio segmented & fully processed imgs.',
    7: '<===Back',
}
CNN_menu_options = {
    1: 'My CNN',
    2: 'LeNet5',
    3: 'EfficientNetB0',
    4: 'All models',  
    5: '<===Back',
}

#display 
def showHairlessExamples():
    print("Images are being displayed \n")
    hairyImg1 = cv.imread(r"C:/Users/yeyit/Documents/FinalProject/ISIC_2019_Training_Input/ISIC_0003056_downsampled.jpg")
    hairyImg2 = cv.imread(r"C:/Users/yeyit/Documents/FinalProject/ISIC_2019_Training_Input/ISIC_0012969_downsampled.jpg")
    hairyImg3 = cv.imread(r"C:/Users/yeyit/Documents/FinalProject/ISIC_2019_Training_Input/ISIC_0069464.jpg")
    
    # plt.imshow(y_test[1103].astype(int))
    # plt.show()
    #show the image processing steps
    hairReplacement.displayDeleteHair(hairyImg1)
    hairReplacement.displayDeleteHair(hairyImg2)
    hairReplacement.displayDeleteHair(hairyImg3)
def print_menu(menu):
    for key in menu.keys():
        print (key, '--', menu[key] )

def myModelSummary():
        print('{:_^40}'.format('Model summary'))

        mycnn=MyCNN.MyCNN()
        mycnn.loadModel("./CNNs/models/MyCNN/trainedModels/myCNN-11-05-2021 Acc-0.783")
        mycnn.getInfo()   

def getDir(training=False):
    
     print('{:_^40}'.format('Process Images'))
     
     #ask the user if the images are in the dafault dir 
     userPref=input('Use default directories? (Yes/No): ')
     if userPref=='No':
         try:
             dataDir =r''+ input('Enter the directory cointaining the training imgs.: ')
         except:
            print('Something went wrong. Try again ... \n')
         try:# get labels and iamges
             groundTruthDir = r''+input('Enter the directory cointaining the labels: ')
             groundTruth = pd.read_csv(groundTruthDir)
             labels=groundTruth.iloc[:,1]
             labels=np.array(labels)
             labels=tf.keras.utils.to_categorical(labels.astype(np.float32), num_classes)
         except:
            print('Something went wrong. Try again ... \n')
     else:
        dataDir="./sampleImgs" 
        groundTruthDir="./data/ISIC_2019_Training_GroundTruth_Sample.csv"
        groundTruth = pd.read_csv(groundTruthDir)
        labels=groundTruth.iloc[:,1]
        labels=np.array(labels)
        labels=tf.keras.utils.to_categorical(labels.astype(np.float32), num_classes)        
    #if the diretiry exists
     print(dataDir)

     processImages(dataDir,labels,option,training)
     
#create small models      
def trainingDemo(imgs,labels):
     [x_train,x_test,y_train,y_test]=train_test_split(imgs,labels,test_size=0.2)
     
     mycnnDemo=MyCNN.MyCNN()
     lenetDemo=LeNet5.LeNet_CNN()
     efNetDemo=efficientNet.EfficientNet_CNN()
     
     mycnnDemo.createModel()
     lenetDemo.createModel()
     efNetDemo.createModel()
     
     option = ''

     if option=='':
        while(True):
            print('{:_^40}'.format('Select model'))
            print_menu(CNN_menu_options)
            try:
                option = int(input('Enter your choice: '))
            except:
                print('Wrong input. Please enter a number ...')
            #Check what choice was entered and act accordingly
            if option == 1:
                mycnnDemo.trainModel(x_train,x_test,y_train,y_test,epochNum=40)
                
            elif option == 2:
                lenetDemo.trainModel(x_train,x_test,y_train,y_test,epochNum=40)
            elif option == 3:
                efNetDemo.trainModel(x_train,x_test,y_train,y_test,epochNum=10)
            elif option==4:#train all of them
                mycnnDemo.trainModel(x_train,x_test,y_train,y_test,epochNum=40)
                lenetDemo.trainModel(x_train,x_test,y_train,y_test,epochNum=80)
                efNetDemo.trainModel(x_train,x_test,y_train,y_test,epochNum=10)
            elif option == 5:
                 print('Returning to previous menu.')
                 break
            else:
                print('Invalid option. Please enter a number between 1 and 5.')
def compareModels(models,x_test,y_test):
    scores=testing.testModels(models,x_test,y_test)        
    graphsGenerator.compareROC(scores,y_test)
    
#predict classes of images saved in ./imgsToPredict    
def predictClass():
        print('Loading previously trained models... \n')
        mycnn=MyCNN.MyCNN()
        mycnn.loadModel("./CNNs/models/MyCNN/trainedModels/myCNN-11-05-2021 Acc-0.783")
        #get lenet5
        lenet=LeNet5.LeNet_CNN()
        lenet.loadModel("./CNNs/models/LeNet/trainedModels/LeNet5-07-05-2021Acc-0.73")
        #get efficientNetb0 
        efNet=efficientNet.EfficientNet_CNN()
        efNet.loadModel("./CNNs/models/EfficientNet/trainedModels/EfficientNetB0-09-05-2021 Acc-0.793")
        predictDir="./imgsToPredict"
        
        print('Processing images... \n')
        [imgs,_,_]=customedProcessing.adjustableProcessing(predictDir,[''],evenRatio=False, 
                                    hairRemoval=True,borderRemoval=True,segmentation=False)
        

        option = ''
        #select the prediction method
        if option=='':
           while(True):
               print('{:_^40}'.format('Select model'))

               print_menu(CNN_menu_options)
               try:
                   option = int(input('Enter your choice: \n'))
               except:
                   print('Wrong input. Please enter a number ... \n')
               #Check what choice was entered and act accordingly
               if option == 1:
                    mycnn_predictions=mycnn.predictImgs(imgs)
               elif option == 2:
                    lenet_predictions=lenet.predictImgs(imgs)

               elif option == 3:
                    efNet_predictions=efNet.predictImgs(imgs)

               elif option==4:
                   
                    mycnn_predictions=mycnn.predictImgs(imgs)
                    lenet_predictions=lenet.predictImgs(imgs)
                    efNet_predictions=efNet.predictImgs(imgs)
                    

               elif option == 5:
                    print('Returning to previous menu./n')
                    break
               else:
                   print('Invalid option. Please enter a number between 1 and 5. \n')
        
        
#test the previously trained models        
def testModel(x_test,y_test):
        print('Loading previously trained models... \n')
        mycnn=MyCNN.MyCNN()
        mycnn.loadModel("./CNNs/models/MyCNN/ myCNN-07-05-2021 Acc-0.78")
        #get lenet5
        lenet=LeNet5.LeNet_CNN()
        lenet.loadModel("./CNNs/models/LeNet/ LeNet5-07-05-2021Acc-0.73")
        #get efficientNetb0 
        efNet=efficientNet.EfficientNet_CNN()
        efNet.loadModel("./CNNs/models/EfficientNet/ EfficientNetB0-09-05-2021 Acc-0.793")
        print('Select Model')
        models=[mycnn,lenet,efNet]
        option = ''
   
        if option=='':
           while(True):
               print('{:_^40}'.format('Select model'))

               print_menu(CNN_menu_options)
               try:
                   option = int(input('Enter your choice: '))
               except:
                   print('Wrong input. Please enter a number ...')
               #Check what choice was entered and act accordingly
               if option == 1:
                   mycnn.showPerformance(x_test,y_test,classes=[1,0])
                   
               elif option == 2:
                   lenet.showPerformance(x_test,y_test,classes=[1,0])
               elif option == 3:
                   efNet.showPerformance(x_test,y_test,classes=[1,0])
               elif option==4:
                   compareModels(models,x_test,y_test)
               elif option == 5:
                    print('Returning to previous menu.')
                    break
               else:
                   print('Invalid option. Please enter a number between 1 and 5.')
#ask the user what way they want to process the images
def processImages(dataDir,labels,option,training):
     print('Select processing option for the validation images (recomended=5)')
     imgs=[]
     if os.path.isdir(dataDir)==True:
        while(True):
            print_menu(processing_menu_options)
            option = ''
            try:
                option = int(input('Enter your choice: '))
            except:
                print('Wrong input. Please enter a number ...')
            #Check what choice was entered and act accordingly
            if option == 1:#no image processing
                [imgs,labels,_]=customedProcessing.adjustableProcessing(dataDir,labels,evenRatio=False, 
                                    hairRemoval=False,borderRemoval=False,segmentation=False)
                if training==False:
                    testModel(imgs,labels)
                else:
                    trainingDemo(imgs,labels)
            elif option == 2:#augmented
                #show how the pocessings methods affect images
                [imgs,labels,_]=customedProcessing.adjustableProcessing(dataDir,labels,evenRatio=True, 
                                    hairRemoval=False,borderRemoval=False,segmentation=False)
                if training==False:
                    testModel(imgs,labels)
                else:
                    trainingDemo(imgs,labels)
            elif option == 3:#augmented borderless
                [imgs,labels,_]=customedProcessing.adjustableProcessing(dataDir,labels,evenRatio=True, 
                                    hairRemoval=False,borderRemoval=True,segmentation=False)
                if training==False:
                    testModel(imgs,labels)
                else:
                    trainingDemo(imgs,labels)
            elif option == 4:#augmented hairless
                [imgs,labels,_]=customedProcessing.adjustableProcessing(dataDir,labels,evenRatio=True, 
                                    hairRemoval=True,borderRemoval=False,segmentation=False)
                if training==False:
                    testModel(imgs,labels)
                else:
                    trainingDemo(imgs,labels)
            elif option == 5:#augmented borderless & hairless
                [imgs,labels,_]=customedProcessing.adjustableProcessing(dataDir,labels,evenRatio=True, 
                                    hairRemoval=True,borderRemoval=True,segmentation=False)
                if training==False:
                    testModel(imgs,labels)
                else:
                    trainingDemo(imgs,labels)
            elif option == 6:#augmented segmented, borderless & hairless
                [imgs,labels,_]=customedProcessing.adjustableProcessing(dataDir,labels,evenRatio=True, 
                                    hairRemoval=True,borderRemoval=True,segmentation=True)
                if training==False:
                    testModel(imgs,labels)
                else:
                    trainingDemo(imgs,labels)
            elif option == 7:
                print('Returning to previous menu.')
                break
            else:
                print('Invalid option. Please enter a number between 1 and 7.')
if __name__=='__main__':
    while(True):
        print('{:_^40}'.format('Main Menu'))
        print_menu(menu_options)
        option = ''
        try:
            option = int(input('Enter your choice: \n'))
        except:
            print('Wrong input. Please enter a number ... \n')
        #Check what choice was entered and act accordingly
        if option == 1:
           showHairlessExamples()
        elif option == 2:
            #show how the pocessings methods affect images
            testing.showImgProUnique10()
            
        elif option == 3:
            myModelSummary()
        elif option == 4:
            getDir()
        elif option == 5:
            getDir(training=True)
        elif option == 6:
            predictClass()
        elif option == 7:
            print('Exiting program')
            break
        else:
            print('Invalid option. Please enter a number between 1 and 4. \n')