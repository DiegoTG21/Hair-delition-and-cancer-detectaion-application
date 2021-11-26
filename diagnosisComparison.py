# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 12:16:07 2021

@author: yeyit
"""

#import neccesary libraries
from progressbar import progressbar
import numpy as np
import tensorflow as tf # tensorflow 2.0 Google Deep learning release
import cv2 as cv
import os
import sys
import imgaug.augmenters as iaa

from sklearn.model_selection import train_test_split#randomises
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.metrics import f1_score

#import my funcs
sys.path.append("./CNNs/functions")
import CNNs.models.MyCNN.MyCNN as MyCNN
import CNNs.models.LeNet.LeNet5 as LeNet
import CNNs.models.MobileNet.MobileNet as MobileNetV2


import customedProcessing

num_classes=2

#Load directiries
dataDirOthers=r".\ISIC_Balanced_Others" 
dataDirMel=r".\isic_cleaned_mel" 

# loading in the data

annoOthers=pd.read_csv(r'.\final annotations\imgInterfernces_ISIC_Balanced_Others_final.csv')
annoMel=pd.read_csv(r'.\final annotations\isic_cleaned_mel_Annotations.csv') 

hairGroundTruth_mel = pd.read_csv(r'.\final annotations\ISIC_hair_anno_final.csv')
hairGroundTruth_others = pd.read_csv(r'.\final annotations\hair_anno_ISIC_Balanced_Others_final.csv')

groundTruth_mel = pd.read_csv(r'.\isic_cleaned_mel_Annotations.csv')
#get labels for hair in mel
gt_mel=groundTruth_mel.iloc[:,2]
gt_mel=gt_mel.to_numpy()

groundTruth_others = pd.read_csv(r'.\final annotations\imgInterfernces_ISIC_Balanced_Others_final.csv')
#get labels for hair in others

gt_others=groundTruth_others.iloc[:,2]
gt_others=gt_others.to_numpy()


#get all the images with hairs and without hairs in 2 different arrays with the corresponding labels
def getData(labels_mel,labels_others,hairGroundTruth_mel,hairGroundTruth_others,lowerBound=1,rotate=False):
    hairlevels=[]

    imgArray=[]
    imgArrayClean=[]
    newLabels=[]
    newLabelsClean=[]

    indeces=[]
    indecesClean=[]

    counter=0
    #loo through every image 
    for file in progressbar(os.listdir(dataDirMel)):
        if hairGroundTruth_mel.iloc[counter,1]==1:
            hairlevels.append(1)
        elif hairGroundTruth_mel.iloc[counter,2]==1:
            hairlevels.append(2) 
        elif hairGroundTruth_mel.iloc[counter,3]==1:
            hairlevels.append(3)
        elif hairGroundTruth_mel.iloc[counter,4]==1:
            hairlevels.append(4)
        elif hairGroundTruth_mel.iloc[counter,5]==1:
            hairlevels.append(5)
        elif hairGroundTruth_mel.iloc[counter,6]==1:
            hairlevels.append(6)
        else:
            hairlevels.append(0)
        #if the image is label as hairless add to clean array
        if annoMel.iloc[counter,2]==0:
            colourImg=cv.resize(cv.imread(os.path.join(dataDirMel,file)), (300,300))#300 is the ideal size for hair removal method 2

            imgArrayClean.append(np.float32(colourImg))
            newLabelsClean.append(labels_mel[counter])
            indecesClean.append(counter)
        #if level 1 and 2 are == 0 and the image has hair it means it is a level 3, 4, 5 or 6
        elif hairGroundTruth_mel.iloc[counter,1]==0 and hairGroundTruth_mel.iloc[counter,2]==0:
            #get colour img
            colourImg=cv.resize(cv.imread(os.path.join(dataDirMel,file)), (300,300))#300 is the ideal size for hair removal method 2
            #random rotate less overfitting
            if rotate==True:
                rotate=iaa.Affine(rotate=(-180, 180))
                rotatedImg=rotate.augment_image(colourImg)
                imgArray.append((rotatedImg))
            else:#add image to array
                imgArray.append(np.float32(colourImg))
            #add labels and indeces
            indeces.append(counter)
            newLabels.append(labels_mel[counter])
        else:#level 1 and 2 are added to clean array
            imgArrayClean.append(np.float32(colourImg))
            newLabelsClean.append(labels_mel[counter])
            indecesClean.append(counter)
            
        counter=counter+1
        
        
    counter=0
    #get images and proccess them so the potential melanomas have more defined borders
    for file in progressbar(os.listdir(dataDirOthers)):
        if hairGroundTruth_others.iloc[counter,1]==1:
            hairlevels.append(1)
        elif hairGroundTruth_others.iloc[counter,2]==1:
            hairlevels.append(2) 
        elif hairGroundTruth_others.iloc[counter,3]==1:
            hairlevels.append(3)
        elif hairGroundTruth_others.iloc[counter,4]==1:
            hairlevels.append(4)
        elif hairGroundTruth_others.iloc[counter,5]==1:
            hairlevels.append(5)
        elif hairGroundTruth_others.iloc[counter,6]==1:
            hairlevels.append(6)
        else:
            hairlevels.append(0)
        if annoOthers.iloc[counter,2]==0:
            colourImg=cv.resize(cv.imread(os.path.join(dataDirOthers,file)), (300,300))#300 is the ideal size for hair removal method 2

            imgArrayClean.append(np.float32(colourImg))
            newLabelsClean.append(labels_others[counter])
            indecesClean.append(counter)
        elif hairGroundTruth_others.iloc[counter,1]==0 and hairGroundTruth_others.iloc[counter,2]==0:
            #get colour img
            colourImg=cv.resize(cv.imread(os.path.join(dataDirOthers,file)), (300,300))#300 is the ideal size for hair removal method 2
            #random rotate less overfitting
            if rotate==True:
                rotate=iaa.Affine(rotate=(-180, 180))
                rotatedImg=rotate.augment_image(colourImg)
                imgArray.append((rotatedImg))
            else:
                imgArray.append(np.float32(colourImg))
            indeces.append(counter)
            newLabels.append(labels_others[counter])

    
        else:
            imgArrayClean.append(np.float32(colourImg))
            newLabelsClean.append(labels_others[counter])
            indecesClean.append(counter)
        counter=counter+1
        
    return imgArray, newLabels, indeces, imgArrayClean, newLabelsClean,indecesClean,hairlevels

#get a set of images and return them processed with each method
def getProcessData(imgArray, labels_categorical):#,hairlevels):
    [hairless_imgs_method1,_,_]= customedProcessing.adjustableImageProcessing(imgArray,labels_categorical,evenRatio=False,# hairLevels=hairlevels,
                            hairRemoval=True,hairRemovalMethod=1,borderRemoval=False,segmentation=False) 
    [hairless_imgs_method2,_,_]= customedProcessing.adjustableImageProcessing(imgArray,labels_categorical,evenRatio=False, # hairLevels=hairlevels,
                                    hairRemoval=True,hairRemovalMethod=2,borderRemoval=False,segmentation=False) 
    [hairless_imgs_method3,_,_]= customedProcessing.adjustableImageProcessing(imgArray,labels_categorical,evenRatio=False,  #hairLevels=hairlevels,
                                    hairRemoval=True,hairRemovalMethod=3,borderRemoval=False,segmentation=False) 
    # [hairless_imgs_method5,_,_]= customedProcessing.adjustableImageProcessing(imgArray,labels_categorical,evenRatio=False,  hairLevels=hairlevels,
    #                                 hairRemoval=True,hairRemovalMethod=4,borderRemoval=False,segmentation=False) 
    return hairless_imgs_method1,hairless_imgs_method2,hairless_imgs_method3#,hairless_imgs_method4

#get images and create new labels for mels. and others
def createLabels(indeces):
    lastIndex=-1
    labels=[]
    diagnosis=1
    for index in indeces:#if last index is higher than the current one the other images have been reached
        if index<lastIndex:
            diagnosis=0
            labels.append(diagnosis)
        else:
            labels.append(diagnosis)
        lastIndex=index
    
    return labels

#get metrics for a model
            
def getModelResult(model,y_test,x_test):
    predictions=model.model.predict_classes(y_test)
    predictions_percent=model.model.predict(y_test)
    # print (predictions[1:5])
    # print (predictions_percent[1:5])

    # totalImgs=len(y_test)
    # correctPre=0
    # tp,fp,tn,fn=0,0,0,0
    # counter =0
    benignAcc=0
    melAcc=0
    conMat = tf.math.confusion_matrix(labels=x_test[:,1], predictions=predictions).numpy()
    conMatNorm = np.around(conMat.astype('float') / conMat.sum(axis=1)[:, np.newaxis], decimals=4)


    sensitivity = conMatNorm[0,0]/(conMatNorm[0,0]+conMatNorm[0,1])
    print('Sensitivity : ', sensitivity)
    
    specificity = conMatNorm[1,1]/(conMatNorm[1,0]+conMatNorm[1,1])
    print('Specificity : ', specificity)
    fpr, tpr, _ = roc_curve(x_test[:,1], predictions_percent[:,1])
    ROC = auc(fpr, tpr)            
    melAcc= conMatNorm[1,1]    
    benignAcc= conMatNorm[0,0]       
    # print('Mel acc.=', melAcc)
    # print('Benign acc=',benignAcc)
    
    f1Score =f1_score(x_test[:,1], predictions, average='binary')

    return melAcc,benignAcc,sensitivity,specificity,ROC,f1Score
def trainModel(imgArray,labels,aug=False, modelType=1):#return best performing model

        cat_labels=tf.keras.utils.to_categorical(labels, num_classes)
        
        if aug==True:
            imgArray,cat_labels,randomIndices=customedProcessing.adjustableImageProcessing(imgArray,cat_labels,evenRatio=aug,
                                   hairRemoval=False,borderRemoval=False,imgSize=128,
                                  segmentation=False)
            print('Num. of images deleted: ',len(randomIndices))
            print('Images used: ', len(cat_labels[:,1]))
            
        #resize imgs
        imgArrayTemp=[]

        for img in imgArray:
                imgArrayTemp.append(cv.resize(img.astype('uint8'), (128,128)))
        imgArrayTemp=np.array(imgArrayTemp)
        imgArray=imgArrayTemp
        [y_train,y_test,x_train,x_test]=train_test_split(
            imgArray,cat_labels,test_size=0.2)
        if modelType==1:
            myCNN=MyCNN.MyCNN()
            myCNN.createModel()
            myCNN.trainModel(y_train,y_test,x_train,x_test,epochNum=30)
           # myCNN.loadCheckpoint()
            return myCNN,y_test,x_test

        elif modelType==2:
            
            leNet=LeNet.LeNet_CNN()
            leNet.createModel()
            leNet.trainModel(y_train,y_test,x_train,x_test,epochNum=45)
            leNet.loadCheckpoint()
            return leNet,y_test,x_test
        elif modelType==3:
            y_train=y_train[:150]
            x_train=x_train[:150]
            MobileNet=MobileNetV2.MobileNetV2_CNN()
            MobileNet.createModel()
            MobileNet.trainModel(y_train,y_test,x_train,x_test,epochNum=20)
            MobileNet.loadCheckpoint()
            return MobileNet,y_test,x_test
def addRow(df,processType,melAcc,benignAcc,sensitivity,specificity,ROC,f1Score,avgAcc):
    new_row = {
	'Type': processType,
	'Mel. accuracy': melAcc,
	'Benign accuracy': benignAcc,
    'Acc':avgAcc,
    'AUC':ROC,
	'Sensitivity': sensitivity,
	'Specificity': specificity,
    'F1 score': f1Score
	}
    #append row to the dataframe
    df = df.append(new_row, ignore_index=True)
    df.to_csv(r'C:\Users\yeyit\OneDrive\Desktop\Hair delition research\diagnosis_results_myCNN_final_metrics_2.csv', index = False)
    return df


#############################
############IGNORE getAllResults()
#############################
def getAllResults(imgArray,hairless_imgs_method1,hairless_imgs_method2,hairless_imgs_method3,hairless_imgs_method4, labels, hairlevels):
    
    allArrays=[imgArray,hairless_imgs_method1,hairless_imgs_method2,hairless_imgs_method3,hairless_imgs_method4]
    accResults=np.array( [[0]*4]*4)
    melAccOg=0
    benignAccOg=0
    data = {	'Type': [],
	'Mel. accuracy': [],
	'Benign accuracy': [],
    'Avg acc':[],
    'ROC':[],
	'Mel. acc. change': [],
	'Benign acc. change': []
	}
    df = pd.DataFrame(data)
    lv3,lv4,lv5,lv6=0,0,0,0
    accuracyPerLevel=0

    counter=0
    models=[]
    for dataSet in allArrays:
        
        if counter==0:
            CNN,y_test,x_test=trainModel(dataSet,labels,aug=True)
            melAcc,benignAcc,ROC,melChange,benignChange=getModelResult(CNN,y_test,x_test,ogImgs=True)
            benignAccOg=benignAcc
            melAccOg=melAcc
            models.append(CNN)
        else:
            CNN,y_test,x_test=trainModel(dataSet,labels,aug=True)
            melAcc,benignAcc,ROC,melChange,benignChange=getModelResult(CNN,y_test,x_test,melAccOg=melAccOg,benignAccOg=benignAccOg)
            models.append(CNN)

        df=addRow(df,counter,melAcc,benignAcc,ROC,melChange,benignChange)
        
        counter=counter+1
    return models,accResults

def testModel(CNN,imgArray,hairless_imgs_method1,hairless_imgs_method2,hairless_imgs_method3, labels):
    allArrays=[imgArray,hairless_imgs_method1,hairless_imgs_method2,hairless_imgs_method3]
    #lenet aug hairless og imgs
    melAccOg=0

    benignAccOg=0

    data = {	'Type': [],
	'Mel. accuracy': [],
	'Benign accuracy': [],
    'Acc':[],
    'AUC':[],
	'Sensitivity': [],
	'Specificity': [],
    'F1 score': []
	}    
    df = pd.DataFrame(data)
    labels=np.array(labels)
    labels=tf.keras.utils.to_categorical(labels, num_classes)
    counter=0
    for dataSet in allArrays:
        hairy_imgs=[]

        for img in dataSet:
                hairy_imgs.append(cv.resize(img.astype('uint8'), (128,128)))
        hairy_imgs=np.array(hairy_imgs)

        print("Training labels shape: " + str(labels.shape))

        print("Training data shape: " + str(hairy_imgs.shape))
        if counter==0:

            melAcc,benignAcc,sensitivity,specificity,ROC,f1Score=getModelResult(CNN,hairy_imgs,labels)
            score=CNN.modelScore(hairy_imgs,labels)

        else:#
            melAcc,benignAcc,sensitivity,specificity,ROC,f1Score=getModelResult(CNN,hairy_imgs,labels)
            score=CNN.modelScore(hairy_imgs,labels)
        df=addRow(df,counter,melAcc,benignAcc,sensitivity,specificity,ROC,f1Score,score[1])
        
        counter=counter+1
    return df

#get data
imgArray, newLabels, indeces,imgArrayClean, newLabelsClean,indecesClean,hairlevels=getData(gt_mel,gt_others,hairGroundTruth_mel,hairGroundTruth_others)
#convert labels to the right type
diagnosis_labels=createLabels(indeces)
diagnosis_clean_labels=createLabels(indecesClean)
#get all the images processed with each method
hairless_imgs_method1,hairless_imgs_method2,hairless_imgs_method3=getProcessData(imgArray, newLabels)#,hairlevels)
#train a model with the clean images
CNN_noAug,y_test,x_test=trainModel(imgArrayClean,diagnosis_clean_labels,aug=False)
#get results and create table
df=testModel(CNN_noAug,imgArray,hairless_imgs_method1,hairless_imgs_method2,hairless_imgs_method3, diagnosis_labels)