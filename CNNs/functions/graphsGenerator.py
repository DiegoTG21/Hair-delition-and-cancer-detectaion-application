# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:59:32 2021

@author: diego
"""
from progressbar import progressbar
import numpy as np
import os
import tensorflow as tf # tensorflow 2.0 Google Deep learning release
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def ageCorrelation():
    #get data
    gt = pd.read_csv(r'C:\Users\yeyit\Documents\FinalProject\ISIC_2019_Training_GroundTruth.csv')
    md = pd.read_csv(r'C:\Users\yeyit\Documents\FinalProject\ISIC_2019_Training_Metadata.csv')
    mergedData=pd.merge(gt, md, on='image', how='inner')
    
    #rename body location
    md = mergedData.rename(columns = {'anatom_site_general_challenge':'body_loc'})
    #remove missing values
    md = mergedData.dropna(axis=0, how = 'any')
    agePositiveRaw = []
    ageTots=mergedData['age_approx'].value_counts()
    for i in range(mergedData.shape[0]):
        try: 
            if mergedData['age_approx'][i] !='':
                if mergedData['MEL'][i] ==1:
                    #add to the melanoma count 
                    agePositiveRaw.append(mergedData['age_approx'][i]) 
        except:
            pass
        
    #get the count
    agePositive=pd.Index(agePositiveRaw).value_counts()
    ageTots=mergedData['age_approx'].value_counts()
    #sort values from youngest to oldest then divide positives by tot number
    agePositive=agePositive.sort_index()
    ageTots=ageTots.sort_index()
    
    probByAge=agePositive.div(ageTots)
    
    #Show distribution and count
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.distplot(agePositiveRaw)
    plt.title('Distribution of age of people having malignant cancer')
    plt.subplot(1,2,2)
    sns.countplot(y = agePositiveRaw)
    plt.ylabel('Age')
    plt.title('Count plot of age of people having malignant cancer')
    plt.show()
    #plot prob vs age
    probByAge.plot()
    plt.ylabel('Prob. of Mel.')
    plt.xlabel('Age')
    plt.show()

def evaluateModel(model,testImgs, testingLabels):# evaluate the model
    score = model.evaluate(testImgs, testingLabels, verbose=1)
    # print performance
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score
def showPrediction(imgs,predictions,note=''):
    counter =0
    for img in imgs:
        if predictions[counter,0]<0.5:
            plt.imshow(cv.cvtColor(img.astype('uint8'), cv.COLOR_BGR2RGB))
            plt.title(note + " pred. Melanoma (Prob = %0.3f) " % predictions[counter,1])
            plt.show()
        else:
            plt.imshow(cv.cvtColor(img.astype('uint8'), cv.COLOR_BGR2RGB))
            plt.title(note + " pred. Benign (Prob = %0.3f)" % predictions[counter,0] )
            plt.show()
        
        counter=counter+1
def confusionMat(predictions,testImgs,testingLabels,classes):
    #generate confusion mat
    conMat = tf.math.confusion_matrix(labels=testingLabels[:,1], predictions=predictions).numpy()#use 2 row in TLs
    
    conMatNorm = np.around(conMat.astype('float') / conMat.sum(axis=1)[:, np.newaxis], decimals=3)
    
    conMatDf = pd.DataFrame(conMatNorm,
                         index =  [0,1], 
                         columns = [0,1])
    
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(conMatDf, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#ROC curve
def ROC(yScores,x_test):
    fpr, tpr, _ = roc_curve(x_test[:,1], yScores[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plot curve

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
def compareROC(multiple_y_Scores,y_test):
    
    
    colours=['navy','red','blue','green','black','orange','pink','yellow']
    counter=1
    plt.figure()

    for yScores in multiple_y_Scores:
        fpr, tpr, _ = roc_curve(y_test[:,1], yScores[:,1])
        roc_auc = auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr, color=colours[counter-1],
        lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        counter=counter+1
    #plot curve

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def modelAnalysis(model,x_test,y_test,classes):
    #generate predictions
    predictions=model.predict(x_test)
    
    confusionMat(predictions,x_test,y_test,classes)
    #evaluateModel(model,y_test, x_test)
    yScores = model.predict_proba(y_test)
    ROC(yScores,x_test,"CNN")