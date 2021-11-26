# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 20:09:01 2021

@author: yeyit
"""


#import neccesary libraries
from progressbar import progressbar
import numpy as np
import tensorflow as tf # tensorflow 2.0 Google Deep learning release
import cv2 as cv
import matplotlib.pyplot as plt
import os
import random
import sys

from sklearn.model_selection import train_test_split#randomises
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten ,BatchNormalization
from keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import load_model

import efficientnet.tfkeras as efn
#import my funcs
sys.path.append("./CNNs/functions")

import hairReplacement
import testing
import customedProcessing
import graphsGenerator

#gowrong
 # input image dimensions
num_classes = 2 # 2 digits
classes=[1,0]
imgSize=100
#img_rows, img_cols = 240, 360 # number of pixels 
# loading in the data
final_test_data=[]
final_test_labels=[]
# #count mels
# mels=groundTruth['MEL']==1
# gtMels=groundTruth[i]
dataDirOthers=r"C:\Users\yeyit\Downloads\ISIC_Balanced_Others" 
dataDirMel=r"C:\Users\yeyit\Downloads\isic_cleaned_mel" 


hairGroundTruth = pd.read_csv(r'C:\Users\yeyit\OneDrive\Desktop\Hair delition research\final annotations\ISIC_hair_anno_final.csv')


def getSelectedHairImgs(labels,hairGroundTruth_Mel,hairGroundTruth_others,lowerBound=1):
    Hairgt=hairGroundTruth.iloc[:,lowerBound]
    imgArray=[]
    newLabels=[]
    deletedImgArray=[]
    deletedLabels=[]
    counter=0
    #get images and proccess them so the potential melanomas have more defined borders
    for file in progressbar(os.listdir(dataDir)):
        if hairGroundTruth.iloc[counter,lowerBound]==0 and hairGroundTruth.iloc[counter,2]==0:
            #get colour img
            colourImg=cv.resize(cv.imread(os.path.join(dataDir,file)), (300,300))#300 is the ideal size for hair removal method 2
            imgArray.append(np.float32(colourImg))
            newLabels.append(labels[counter])


        counter=counter+1
        
    return imgArray, newLabels
#set the level to what the algo should consider a hair or not
def defineHairs(groundTruth,lowerBound=1, upperBound=6):
    hairGroundTruth = pd.read_csv(r'C:\Users\yeyit\OneDrive\Desktop\Hair delition research\final annotations\ISIC_hair_anno_final.csv')
    gt=groundTruth.iloc[:,2]
    Hairgt_level_one=hairGroundTruth.iloc[:,lowerBound]
    Hairgt_level_two=hairGroundTruth.iloc[:,lowerBound+1]
    gt=gt.to_numpy()
    Hairgt_level_one=Hairgt_level_one.to_numpy()
    Hairgt_level_two=Hairgt_level_two.to_numpy()

    counter=0
    for hairT in Hairgt_level_one:
        if int(hairT)==1:
            gt[counter]=0
        counter=counter+1
        
    counter=0
    for hairT in Hairgt_level_two:
        if int(hairT)==1:
            gt[counter]=0
        counter=counter+1 
    return gt
def getBestProcessData():
    # loading in the data
    groundTruth = pd.read_csv(r'C:\Users\yeyit\OneDrive\Desktop\Hair delition research\final annotations\isic_cleaned_mel_Annotations.csv')
    
    gt=groundTruth.iloc[:,2]
    gt=gt.to_numpy()
    imgArray, labels=getSelectedHairImgs(gt,hairGroundTruth)
    #########USE DEFINEHAIRS()##########
    #labels=defineHairs(groundTruth)
    #extract the label  column
    #labels=groundTruth.iloc[:,2]
    labels_categorical = tf.keras.utils.to_categorical(labels, num_classes)
    [imgArray,labels,randomIndices]= customedProcessing.adjustableImageProcessing(imgArray,labels_categorical,evenRatio=True, 
                                    hairRemoval=False,borderRemoval=False,segmentation=False)
   
    #labels_categorical = tf.keras.utils.to_categorical(labels, num_classes)
    
    
    [x_train,x_test,y_train,y_test]=train_test_split(imgArray,labels,test_size=0.2)
    return [y_train,y_test,x_train,x_test,randomIndices]

def getTestingRawData():
    # loading in the data
    groundTruth = pd.read_csv(r'C:\Users\yeyit\Documents\FinalProject\ISIC_2019_Training_GroundTruth.csv')
    gt=groundTruth.iloc[:,0:2]
    gt=gt.to_numpy()
    dataDir=r"C:\Users\yeyit\Documents\FinalProject\ISIC_2019_Training_Input"
    files = os.listdir(dataDir)
    imgs=[]
    labels=[]
    
    total=0
    numMels=0
    
    while(total<1000):
        index = random.randrange(0, len(files))
        colourImg=cv.imread(os.path.join(dataDir,files[index]))
        if numMels<(total*0.45):
            if gt[index,1]==1:
                colourImg= cv.resize(colourImg, (imgSize,imgSize))
                imgs.append(colourImg)
                #extract the label  column
                labels.append(gt[index,1])
                total=total+1
                numMels=numMels+1
        else:
            colourImg= cv.resize(colourImg, (imgSize,imgSize))
            imgs.append(colourImg)
            #extract the label  column
            labels.append(gt[index,1])
            total=total+1
            if gt[index,1]==1:
                numMels=numMels+1
    
    print(total, numMels)
    #convert to array(easier to handle)
    imgs=np.array(imgs)
    
    #convert to categorical
    labels_categorical = tf.keras.utils.to_categorical(labels, num_classes)
    
    
    return [imgs,labels_categorical]
#models=[]
#display all images with corresponding label
def showImgWithLabel(imgs,label):
    counter=0
    for img in imgs:
        plt.imshow(cv.cvtColor(img.astype('uint8'), cv.COLOR_BGR2RGB))
        plt.title(label[counter][0])
        plt.show()
        counter=counter+1
#CNN=LeNet.CNN(epochNum=50)
# myCNN=MyCNN.CNN()
# myCNN.createModel()
# myCNN.trainModel(x_train,x_test,y_train,y_test,epochNum=30)
# myCNN.showPerformance(x_train,x_test,y_train,y_test)
# models.append(["MyCNN",CNNLoaded])



# lenet_x_train=x_train[:10000]
# lenet_x_test=x_test[:3000]
# lenet_y_train=y_train[:10000]
# lenet_y_test=y_test[:3000]

# full_x_train=customedProcessing.adjustableImageProcessing(x_train,x_test,evenRatio=False, 
#                                     hairRemoval=False,borderRemoval=False,segmentation=True)
# full_x_test=customedProcessing.adjustableImageProcessing(x_train,x_test,evenRatio=False, 
#                                     hairRemoval=False,borderRemoval=False,segmentation=True)
# leNet=LeNet.LeNet_CNN()
# leNet.trainModel(full_x_train,full_x_test,lenet_y_train,lenet_y_test,epochNum=30)

# myCnn=myCNN.CNN()
# myCnn.createModel()
# myCnn.trainModel(full_x_train,full_x_test,lenet_y_train,lenet_y_test,epochNum=30)
# models.append(["LeNet",LeNet])

# EffNet_x_train=x_train[:5000]
# EffNet_x_test=x_test[:1500]
# EffNet_y_train=y_train[:5000]
# EffNet_y_test=y_test[:1500]
# VGG=VGG1080p.VGG1080p_CNN(epochNum=30)
# VGG.trainModel(x_train,x_test,y_train,y_test)
# models.append(["VGG",VGG])
# # CNNLoaded=CNNv6.CNN(data_shape=x_train.shape[1:])
# # CNNLoaded.loadModel()
# # CNN.showPerformance(x_train,x_test,y_train,y_test)
# # models.append(modelv6)


# EffNet_x_train=y_train[:5000]
# EffNet_x_test=y_test[:2000]
# EffNet_y_train=x_train[:5000]
# EffNet_y_test=x_test[:2000]
# en=efficientNet.EfficientNet_CNN()
# en.createModel()

# en.trainModel(EffNet_x_train,EffNet_x_test,EffNet_y_train,EffNet_y_test,epochNum=30)
# import LeNet
# mc.saveModel()

# import LeNet5
# lenet256=LeNet5.LeNet_CNN()
# lenet256.createModel()
# lenet256.trainModel(y_train,y_test,x_train,x_test,epochNum=30)
epochMax=30
#           [x_train,x_test,y_train,y_test,deletedImgsIndeces]=getBestProcessData()
import CNNs.models.MyCNN.MyCNN as MyCNN
#CNN=LeNet.CNN(epochNum=50)
myCNN=MyCNN.MyCNN()
myCNN.createModel()
myCNN.trainModel(x_train,x_test,y_train,y_test,epochNum=30)
mc.saveModel()
# import efficientNet
# en=efficientNet.EfficientNet_CNN()
# en.createModel()

# en.trainModel(EffNet_x_train,EffNet_x_test,EffNet_y_train,EffNet_y_test,epochNum=3)
def imagesFirstLevels(hair_gt):
    counter=0
    newLabels=[]
    imgArray=[]
    #get images and proccess them so the potential melanomas have more defined borders
    for file in progressbar(os.listdir(dataDir)):
        if hair_gt.iloc[counter,1]==1 or hair_gt.iloc[counter,2]==1:
            #get colour img
            colourImg=cv.resize(cv.imread(os.path.join(dataDir,file)), (300,300))#300 is the ideal size for hair removal method 2
            imgArray.append(np.float32(colourImg))
            newLabels.append(1)


        counter=counter+1
    imgArray  =  np.array(imgArray) 
    return imgArray

barelyHairyImgs=imagesFirstLevels(hairGroundTruth)

[hairless_y_test_method3,labels,randomIndices]= customedProcessing.adjustableImageProcessing(y_test,labels_categorical,evenRatio=False, 
                                    hairRemoval=True,borderRemoval=False,segmentation=False)