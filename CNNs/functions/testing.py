# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:06:56 2021

@author: diego 
"""
#import neccesary libraries
import numpy as np
import tensorflow as tf # tensorflow 2.0 Google Deep learning release
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from progressbar import progressbar
import numpy as np
import random
import os
import tensorflow as tf # tensorflow 2.0 Google Deep learning release
import cv2 as cv
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank

try:
    from . import hairReplacement
    from . import customedProcessing
except:
    import hairReplacement
    import customedProcessing
def testModels(models,testImgs,testLabels):
    multiple_y_Scores=[]
    for model in progressbar(models):
        #reshaoe images 
        _,testImgs=model.reshapeImgs(testImgs,testImgs)
        predictions = model.model.predict(testImgs)
        multiple_y_Scores.append(predictions)
        
    return multiple_y_Scores
#show the test acc. and test loss
def evaluateModel(model,testImgs, testingLabels):# evaluate the model
    score = model.evaluate(testImgs, testingLabels, verbose=1)
    # print performance
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score
#display a confusion matrix
def confusionMat(predictions,testImgs,testingLabels,classes,dp=2):
    #generate confusion mat
    conMat = tf.math.confusion_matrix(labels=testingLabels[:,1], predictions=predictions).numpy()#use 2 row in TLs
    
    conMatNorm = np.around(conMat.astype('float') / conMat.sum(axis=1)[:, np.newaxis], decimals=dp)
    
    conMatDf = pd.DataFrame(conMatNorm,
                          index =  [0,1], 
                          columns = [0,1])
    
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(conMatDf, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#display the segmentation process applied to 10 unique images
def showImgProUnique10():
    # Set random seed for purposes of reproducibility
    seed=0
    np.random.seed(seed) # fix random seed
    tf.random.set_seed(seed)
    dataDir= r'C:\Users\yeyit\OneDrive\Desktop\project\projectModel\projectModel\unique10'
    print("hello")
    #declare array
    #imgArray=[]
    #colour arry
    imgArray=[]
    hairlessImgs=[]
    borderlessImgs=[]
    segmentedImgs=[]

    ###counter=1
        #get images
    for file in progressbar(os.listdir(dataDir)):
        #get colour img
        colourImg=cv.imread(os.path.join(dataDir,file))
        imgArray.append(colourImg)
       
        
       #display original
    fig = plt.figure(figsize=(30,10))
    for i in range(len(imgArray)):
        fig.add_subplot(2,5,i+1),plt.imshow(cv.cvtColor(imgArray[i], cv.COLOR_BGR2RGB))
        plt.xticks([]),plt.yticks([])
    plt.show()
    
    
    #display hairless
    for img in progressbar(imgArray):
        #get colour img
        img=hairReplacement.deleteHair(img)
        hairlessImgs.append(img)
    fig = plt.figure(figsize=(30,10))
    for i in range(len(hairlessImgs)):
        fig.add_subplot(2,5,i+1),plt.imshow(cv.cvtColor(hairlessImgs[i], cv.COLOR_BGR2RGB))
        plt.xticks([]),plt.yticks([])
    plt.show()
    
    
    #display borderless
    for cleanImg in progressbar(hairlessImgs):
        borderlessImg=customedProcessing.BorderRemoval(cleanImg)
        borderlessImgs.append(borderlessImg)
    fig = plt.figure(figsize=(30,10))
    for i in range(len(borderlessImgs)):
        fig.add_subplot(2,5,i+1),plt.imshow(cv.cvtColor(borderlessImgs[i], cv.COLOR_BGR2RGB))
        plt.xticks([]),plt.yticks([])
    plt.show()
    
    
    #display final segmented
    for processedImg in progressbar(borderlessImgs):
        processedImg= cv.resize(processedImg, (100,100))
        finalImg=customedProcessing.Segmentation(processedImg)
        segmentedImgs.append(finalImg)
    fig = plt.figure(figsize=(30,10))
    for i in range(len(segmentedImgs)):
        fig.add_subplot(2,5,i+1),plt.imshow(cv.cvtColor(segmentedImgs[i], cv.COLOR_BGR2RGB))
        plt.xticks([]),plt.yticks([])
    plt.show()
    
    
    
    return imgArray#,imgArray
