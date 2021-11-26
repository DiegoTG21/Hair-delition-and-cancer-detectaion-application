# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:26:20 2021

@author: diego
"""

#import neccesary libraries
from progressbar import progressbar
import numpy as np
import random
import os
import tensorflow as tf # tensorflow 2.0 Google Deep learning release
import cv2 as cv
import imgaug.augmenters as iaa
import skimage
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.filters import try_all_threshold, threshold_minimum, threshold_mean
try:
    from . import hairReplacement
except:
    import hairReplacement
imageSize=128
seed=0
np.random.seed(seed) # fix random seed
tf.random.set_seed(seed)

kernel_size_for_edges=int(imageSize/10)#has to be odd

#allows the user to change adjust their desired processing method 
######Takes a directory as input########
def adjustableProcessing(dataDir,labels,evenRatio=True, hairRemoval=True,hairRemovalMethod=3,borderRemoval=True,onlyBorderless=False,
                              segmentation=True,SegmentationMethod="otsu"):
    
    if borderRemoval==True and onlyBorderless==True:
        raise ValueError("Cannot have borderRemoval and onlyBorderless both set as True")
    randomIndices=[]
    counter=0
    melNum=0
    #declare array
    #imgArray=[]
    #colour arry
    imgArray=[]
            #get images and proccess them so the potential melanomas have more defined borders
    for file in progressbar(os.listdir(dataDir)):
        #get colour img
        colourImg=cv.resize(cv.imread(os.path.join(dataDir,file)), (300,300))#300 is the ideal size for hair removal method 2
        
        if hairRemoval==True:
            colourImg=hairReplacement.deleteHair(colourImg,method=hairRemovalMethod)
            
        colourImg= cv.resize(colourImg, (128,128))
            
        if borderRemoval==True:
            colourImg=BorderRemoval(colourImg)
        if onlyBorderless==True:
            if findBorders(colourImg)==True:#do this after hair removal so hairs do not count
            #remove labels and skip any following steps
                 labels=np.delete(labels, (counter), axis=0)
        #reshape img
        if segmentation==True:
            colourImg=Segmentation(colourImg,SegmentationMethod)
         
        #aka augmentation
        if evenRatio==True:
            #if the prediction is benign do not augment
            if labels[counter,1]==0:
                    
                imgArray.append(np.float32(colourImg))
                counter=counter+1
            else:#duplicate and rotate twice and add to array
                #append original img
                imgArray.append(np.float32(colourImg))
                #append flipped and mirrored img
                imgArray.append(np.float32( cv.flip(colourImg, -1)))
                #append 90 deg. rotated img
                imgArray.append(np.float32(np.rot90(colourImg,axes=(0,1))))
                #append rotated and zoomed in img
                rotate=iaa.Affine(rotate=(-20, 20))
                rotatedImg=rotate.augment_image(colourImg)
                crop = iaa.Crop(percent=(0, 0.15)) # crop image (zoom in)
                imgArray.append(np.float32(crop.augment_image(rotatedImg)))
                #duplicate lables 3 times
                labels=np.insert(labels,counter,labels[counter],axis=0)
                labels=np.insert(labels,counter,labels[counter],axis=0)
                labels=np.insert(labels,counter,labels[counter],axis=0)

                counter=counter+4
                #keep track of the num of melanomas
                melNum=melNum+4
        else:
            imgArray.append(np.float32(colourImg))
    if evenRatio==True:
        #convert list to array
        imgArray=np.array(imgArray)
        nonMelIndex=np.where(labels[:,1]==0)
        nonMelIndex=np.asarray(nonMelIndex)
         #convert to 1d array and get the images with such index
        nonMelIndex=nonMelIndex[0,:]
        numOfExpendableImgs=(len(nonMelIndex)-melNum)
        print ("Number of non Mel. imgs: " + str(len(nonMelIndex)))
    
        print ("Number of non Mel. deleted to make a 1:1 ratio: " + str(numOfExpendableImgs))
        if numOfExpendableImgs >0:
            randomIndices=random.sample((nonMelIndex.tolist()),numOfExpendableImgs)# random (delete)
            #delete random non mel imgs
            labels=np.delete(labels, (randomIndices), axis=0)
            imgArray=np.delete(imgArray, (randomIndices), axis=0)
    else:
        #convert list to array
        imgArray=np.array(imgArray)
        labels=np.array(labels)

    return imgArray,labels,randomIndices#,imgArray

    
#allows the user to change adjust their desired processing method 
######Takes an array of images as input########
def adjustableImageProcessing(imgs,labels,evenRatio=True, hairRemoval=True,hairRemovalMethod=3,hairLevels=None,
                              removalLowerBound=2,borderRemoval=True,
                              segmentation=True,imgSize=128,SegmentationMethod="otsu"):
    randomIndices=[]
    counter=0
    melNum=0
    noise_modes = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']  
    noise_mode = noise_modes[random.randint(0, 6)]
    #declare array
    #imgArray=[]
    #colour arry
    imgArray=[]
    imgCounter=0
        #get images and proccess them so the potential melanomas have more defined borders
    for img in progressbar(imgs):
        colourImg=img
        if hairRemoval==True:
            #only remove if there are hairs
            if hairLevels==None:
                
                colourImg=hairReplacement.deleteHair(colourImg,method=hairRemovalMethod)

            elif hairLevels[imgCounter]>removalLowerBound:#only delete hairs if image contains hairs
                colourImg=hairReplacement.deleteHair(colourImg,method=hairRemovalMethod)

        if borderRemoval==True:
            colourImg=BorderRemoval(colourImg)
            
        #reshape img
        colourImg= cv.resize(colourImg, (imgSize,imgSize))
        if segmentation==True:
            colourImg=Segmentation(colourImg,SegmentationMethod)
        #Augmentation            
        if evenRatio==True:
            # if labels[counter,1]==0:
            #     imgArray.append(np.float32(colourImg))
            #     counter=counter+1
            # else:#duplicate and rotate twice and add to array
                #append og img
                imgArray.append(np.float32(colourImg))
                #append flipped and mirrored img
                #imgArray.append(np.float32( cv.flip(colourImg, -1)))
                #append 90 deg. rotated img cropped by 8%
                crop = iaa.Crop(percent=(0, 0.08)) # crop image
                croppedImg=crop.augment_image(colourImg)
                imgArray.append(np.float32(np.rot90(croppedImg,axes=(0,1))))
                #append rotated and zoomed in img
                # rotate=iaa.Affine(rotate=(-20, 20))
                # rotatedImg=rotate.augment_image(colourImg)
                # crop = iaa.Crop(percent=(0, 0.1)) # crop image
                # imgArray.append(np.float32(crop.augment_image(rotatedImg)))
                #add image with noise
                noisyImg = skimage.util.random_noise(colourImg, mode=noise_mode)
                rotate=iaa.Affine(rotate=(-180, 180))
                rotatedNoisyImg=rotate.augment_image(noisyImg)
                imgArray.append(np.float32(np.rot90(rotatedNoisyImg,axes=(0,1))))
                #duplicate groundtruth 3 times
                labels=np.insert(labels,counter,labels[counter],axis=0)
                labels=np.insert(labels,counter,labels[counter],axis=0)
                #labels=np.insert(labels,counter,labels[counter],axis=0)
                #labels=np.insert(labels,counter,labels[counter],axis=0)

                
                ###Change +3 if more images are added when augmented
                counter=counter+3
                #keep track of the num of melanomas
                melNum=melNum+3
        else:
            imgArray.append(np.float32(colourImg))
        imgCounter=imgCounter+1
    if evenRatio==True:
        #convert list to array
        imgArray=np.array(imgArray)
        melIndex=np.where(labels[:,1]==1)
        melIndex=np.asarray(melIndex)
         #convert to 1d array and get the images with such index
        melIndex=melIndex[0,:]
        
        
        nonMelIndex=np.where(labels[:,1]==0)
        nonMelIndex=np.asarray(nonMelIndex)
         #convert to 1d array and get the images with such index
        nonMelIndex=nonMelIndex[0,:]
        numOfExpendableImgs=(len(nonMelIndex)-len(melIndex))
        print("Number of melanoma pictures: %s" %len(melIndex))
        print ("Number of non Mel. imgs: " + str(len(nonMelIndex)))
    
        print ("Number of non Mel. deleted to make a 1:1 ratio: " + str(numOfExpendableImgs))
        if len(melIndex)<len(nonMelIndex):
            randomIndices=random.sample((nonMelIndex.tolist()),numOfExpendableImgs)# 6k random (delete)
        else:
            randomIndices=random.sample((melIndex.tolist()),numOfExpendableImgs*(-1))# 6k random (delete)

        #delete random non mel imgs
        labels=np.delete(labels, (randomIndices), axis=0)
        imgArray=np.delete(imgArray, (randomIndices), axis=0)
    else:
        #convert list to array
        imgArray=np.array(imgArray)
        labels=np.array(labels)

    return imgArray,labels,randomIndices#,imgArray
#use otsu to segment
def Segmentation(colourImg,method='otsu'):
        #black and white image
        colourImg=colourImg.astype('uint8')
        grayImg = cv.cvtColor(colourImg, cv.COLOR_BGR2GRAY)
        #otsu threshold removes backgroud and inter. like hair
        if method=='otsu':
            radius = 3
            selem = disk(radius)
            #apply otsu to img
            local_otsu = rank.otsu(grayImg, selem)
            threshold_global_otsu = threshold_otsu(grayImg)
            #use the global threshold to remove pixels that are not dark enough
            global_otsu = local_otsu >= threshold_global_otsu
            # mean_thresh = threshold_mean(grayImg)
            # #use the global threshold to remove pixels that are not dark enough
            # global_otsu = local_otsu >= mean_thresh
            #remove background from colour image
            colourImg[global_otsu[:,:]==True,:]=255
        else:
            print("No other methods avaliable at the moment")
        return colourImg
    
    #remove most part of the darkest borders which enhancest the segmenting process 
def BorderRemoval(colourImg):
        blur = cv.GaussianBlur(colourImg, (5,5),2)#helps conserving dark values in lesions
        black=np.array([3,3,3])
        white=np.array([255,255,255])
        #conserve all the values lighter than rgb(3,3,3)
        mask = cv.inRange(blur, black, white)
        #trim edges 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(kernel_size_for_edges),int(kernel_size_for_edges)))
        mask = cv.dilate(cv.bitwise_not(mask.astype('uint8')), kernel)
        mask =cv.bitwise_not(mask)

     
        if np.any(mask==0):

             #replace dark pixels with with pixels
             colourImg[mask[:,:]==0,:]=255
            # colourImg = cv.inpaint(colourImg, mask, inpaintRadius=25, flags=cv.INPAINT_TELEA)
        return colourImg
def findBorders(colourImg):
        blur = cv.GaussianBlur(colourImg, (5,5),2)#helps conserving small dark values in lesions
        black=np.array([3,3,3])
        white=np.array([255,255,255])
        #conserve all the values lighter than rgb(3,3,3)
        mask = cv.inRange(blur, black, white)
        #do not return an image if there are black values
        if np.any(mask==0):
            return True
        return False