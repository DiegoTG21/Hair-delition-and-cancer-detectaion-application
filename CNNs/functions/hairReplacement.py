# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:29:21 2021

@author: diego
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_minimum, threshold_mean
from skimage.filters import threshold_otsu, rank, threshold_local
from skimage.util import img_as_ubyte
imgSize_method_one=450
imgSize_method_two=300
block_size_method_one = 13
block_size_method_two = 9
inpaintRadius=4

##############################CODE WAS DELETED AS WE ARE STILL WORKING ON RESEARCH##################################

#########IGNORE###########
def displayDeleteHair(img,method=3):
    processImgs=[]
    font = {
        'size'   : 18}
    plt.rc('font', **font)
    #get and resize img and create a colour copy
        #use the right parameters for each method
    if method==2 or method == 3 or method ==4:
        imgSize=imgSize_method_two
        block_size=block_size_method_two
    elif method==1:
        imgSize=imgSize_method_one
        block_size=block_size_method_one
    else:
        raise ValueError("Only 3 options for hair removal")

    img= cv.resize(img, (imgSize,imgSize))
    colourImg= cv.resize(img, (imgSize,imgSize))
    processImgs.append(cv.cvtColor(np.array(colourImg), cv.COLOR_BGR2RGB))
    #convert to grayscale to get threshold
    grayImg=cv.cvtColor(colourImg,cv.COLOR_BGR2GRAY)

                
    #gray scale 
    imgTemp=grayImg
    
    # #use adaptive thr to enhanced edges and hair
    adaptive_thresh = threshold_local(imgTemp, block_size, offset=10)
    binary_adaptive = imgTemp > adaptive_thresh
    processImgs.append(np.array(binary_adaptive))

    if method==1:
        thresh_adapt = cv.adaptiveThreshold(grayImg, 255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51,1 )
        processImgs.append(cv.cvtColor(np.array(thresh_adapt), cv.COLOR_BGR2RGB))

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        processImgs.append(np.array(thickerHairs))
        
        #get hairs
        # binary_adaptive[thresh_adapt[:,:]==255]=255
        # processImgs.append(np.array(binary_adaptive))
        
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerHairs[:,:]==255]=1
        mask=mask.astype('uint8')
        #img[mask[:,:]==1,:]=255
        inpainted_img = cv.inpaint(img, mask, inpaintRadius=6, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thresh_adapt[:,:]==0,:]
        

        mask2=np.copy(img)
        mask2[:,:,:]=0
        mask2[thresh_adapt[:,:]==0,:]=hair_implants
        processImgs.append(cv.cvtColor(mask2.astype('uint8'), cv.COLOR_BGR2RGB))

        #processImgs.append(cv.cvtColor(np.array(inpainted_img), cv.COLOR_BGR2RGB))
        #replace with the hairs
        img[thresh_adapt[:,:]==0,:]=hair_implants
        
        processImgs.append(cv.cvtColor(np.array(img), cv.COLOR_BGR2RGB))
        titles = ['Original Image','Binary adaptive  (block size=%d)' %block_size,
                  'Adaptive thr.',
                  
                        'Thicken inpainted hairs', 
                        'Area replaced', 'Final image']
        fig = plt.figure(figsize=(20,10))
        for i in range(len(processImgs)):
            fig.add_subplot(2,3,i+1),plt.imshow((processImgs[i]))
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
            plt.axis('off')

        plt.show()
        #plt.title("Final ")
        plt.show()
        return img
    elif method==2:
    
       #binary_adaptive_blur = cv.medianBlur(binary_adaptive.astype('uint8'),3)
       #thicken hairs before inpainting
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        processImgs.append(np.array(thickerHairs))
        
        #get slightly thicker hairs
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thickerImplants = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        
        processImgs.append(np.array(thickerImplants))
    
        #get hair 
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerHairs[:,:]==255]=1
        mask=mask.astype('uint8')

        inpainted_img = cv.inpaint(img, mask, inpaintRadius=inpaintRadius, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thickerImplants[:,:]==255,:]
        
        processImgs.append(cv.cvtColor(np.array(inpainted_img), cv.COLOR_BGR2RGB))
        #replace with the thinner hairs
        img[thickerImplants[:,:]==255,:]=hair_implants
        
        processImgs.append(cv.cvtColor(np.array(img), cv.COLOR_BGR2RGB))
         
        titles = ['Original Image', 'Binary adaptive  (block size=%d)' %block_size,
                        'Hairs for inpainting', 'Replacement hairs',
                        'Inpainted img using 5x5 thicker hairs', 'Inpainted hairs using 3x3']
        fig = plt.figure(figsize=(20,10))
        for i in range(len(processImgs)):
            fig.add_subplot(2,3,i+1),plt.imshow((processImgs[i]))
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
            plt.axis('off')

        #plt.title("Final ")
        plt.show()
        return img
    else:
        plt.imshow(binary_adaptive)
        plt.title("thresh without edge")
        plt.axis('off')
        #processImgs.append(binary_adaptive)

        plt.show()
        otsu_thresh= threshold_otsu(cv.cvtColor(colourImg,cv.COLOR_BGR2GRAY))
        edges = cv.Canny(colourImg,otsu_thresh,160)
        plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))
        plt.title("Edges used for enhancement")
        plt.axis('off')
        processImgs.append(cv.cvtColor(edges, cv.COLOR_BGR2RGB))

        plt.show()   
        grayImg[edges[:,:]==255]=0
        plt.imshow(cv.cvtColor(grayImg, cv.COLOR_BGR2RGB))
        plt.title("Enhanced gray image")
        plt.axis('off')
        #processImgs.append(cv.cvtColor(grayImg, cv.COLOR_BGR2RGB))

        plt.show()
        binary_adaptive = grayImg > adaptive_thresh
        plt.imshow(binary_adaptive)
        plt.title("thresh with edge")
        plt.axis('off')
        processImgs.append(binary_adaptive)

        plt.show()
        #make hairs thicker to cover shadows
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        
        #get slightly thiner hairs
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thickerImplants = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        # binary_adaptive[thresh_adapt[:,:]==255]=255
        # plt.imshow(thickerHairs)
        # plt.show()
        # plt.imshow(thickerImplants)
        # plt.show()
        #create a mask using the thick hairs
    
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerImplants[:,:]==255]=1
        #mask[edges[:,:]==255]=1
        mask=mask.astype('uint8')
        # plt.imshow(mask)
        # plt.show()
        #img[mask[:,:]==1,:]=255
        img=img.astype('uint8')
        #inpaint thick hairs
        inpainted_img = cv.inpaint(img, mask, inpaintRadius=4, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thickerImplants[:,:]==255,:]
        processImgs.append(thickerImplants)

    
        #replace with the thinner hairs
        img[thickerImplants[:,:]==255,:]=hair_implants
        #return clean image
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.axis('off')
        processImgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title("Method 3")
        titles = ['Original Image', 'Binary adaptive  (block size=%d)' %block_size,
                        'Edges used for enhancement', 'Thresh. with Canny edge enhancement',
                        'Area to be inpainted', 'Final image']
        fig = plt.figure(figsize=(20,10))
        for i in range(len(processImgs)):
            fig.add_subplot(2,3,i+1),plt.imshow((processImgs[i]))
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
            plt.axis('off')

        #plt.title("Final ")
        plt.show()
        # plt.show()
        # titles = ['Original Image', 'Binary adaptive  (block size=%d)' %block_size,
        #                 'Thicken hairs using 5x5 kernel', 'Thicken hairs using 3x3 kernel',
        #                 'Inpainted img using 5x5 thicker hairs', 'Inpainted hairs using 3x3']
        # fig = plt.figure(figsize=(20,10))
        # for i in range(len(processImgs)):
        #     fig.add_subplot(2,3,i+1),plt.imshow((processImgs[i]))
        #     plt.title(titles[i])
        #     plt.xticks([]),plt.yticks([])
        # plt.show()
        return img
def deleteHair(img,method=1):
    #get and resize img and create a colour copy
        #use the right parameters for each method
    if method==2 or method == 3 or method ==4:
        imgSize=imgSize_method_two
        block_size=block_size_method_two
    elif method==1 or method ==5:
        imgSize=imgSize_method_one
        block_size=block_size_method_one
    else:
        raise ValueError("Only 5 options for hair removal")
        
    img=img.astype('uint8')
    img= cv.resize(img, (imgSize,imgSize))
    colourImg= img
    #convert to grayscale to get threshold
    grayImg=cv.cvtColor(colourImg,cv.COLOR_BGR2GRAY)

 


    # #use adaptive thr to enhanced edges and hair
    adaptive_thresh = threshold_local(grayImg, block_size, offset=10)
    binary_adaptive = grayImg > adaptive_thresh
    

    if method==1:
        grayImg=grayImg.astype('uint8')

    #use adaptive threshold to find hairs and wanted areas
        thresh_adapt = cv.adaptiveThreshold(grayImg, 255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51,1 )

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)        
        #get hairs
        # binary_adaptive[thresh_adapt[:,:]==255]=255
        # processImgs.append(np.array(binary_adaptive))
        
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerHairs[:,:]==255]=1
        mask=mask.astype('uint8')
        #img[mask[:,:]==1,:]=255
        img=img.astype('uint8')

        inpainted_img = cv.inpaint(img, mask, inpaintRadius=6, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thresh_adapt[:,:]==0,:]
        
        #replace with the hairs
        img[thresh_adapt[:,:]==0,:]=hair_implants
        
        return img
    elif method==2:
   
        #make hairs thicker to cover shadows
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        
        #get slightly thiner hairs
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thickerImplants = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        # binary_adaptive[thresh_adapt[:,:]==255]=255
    
        #create a mask using the thick hairs
    
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerHairs[:,:]==255]=1
        mask=mask.astype('uint8')
        #img[mask[:,:]==1,:]=255
        img=img.astype('uint8')
        #inpaint thick hairs
        inpainted_img = cv.inpaint(img, mask, inpaintRadius=inpaintRadius, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thickerImplants[:,:]==255,:]
        
    
        #replace with the thinner hairs
        img[thickerImplants[:,:]==255,:]=hair_implants
        #return clean image
        return img
    elif method ==3 :
        otsu_thresh= threshold_otsu(cv.cvtColor(colourImg,cv.COLOR_BGR2GRAY))
        #get edges using canny's method, otsu thresholding and an upper thresh limit(set by user)
        edges = cv.Canny(colourImg,otsu_thresh,160)
        #enhance the gray image with the edges
        grayImg[edges[:,:]==255]=0
        #find hairs using binary adaptive
        binary_adaptive = grayImg > adaptive_thresh
        
        #make hairs thicker to cover shadows
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        
        #get slightly thiner hairs
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thickerImplants = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)
        # binary_adaptive[thresh_adapt[:,:]==255]=255
        # plt.imshow(thickerHairs)
        # plt.show()
        # plt.imshow(thickerImplants)
        # plt.show()
        #create a mask using the thick hairs
    
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerImplants[:,:]==255]=1
        #mask[edges[:,:]==255]=1
        mask=mask.astype('uint8')
        # plt.imshow(mask)
        # plt.show()
        #img[mask[:,:]==1,:]=255
        img=img.astype('uint8')
        #inpaint thick hairs
        inpainted_img = cv.inpaint(img, mask, inpaintRadius=4, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thickerImplants[:,:]==255,:]
        
    
        #replace with the slightly thick hairs
        img[thickerImplants[:,:]==255,:]=hair_implants
        #return clean image

        return img
    elif method==4:
        
        imgSize=300
        #get and resize img and create a colour copy
        img= cv.resize(img, (imgSize,imgSize))
        colourImg= img
    
        #convert to grayscale to get threshold
        grayImg=cv.cvtColor(colourImg,cv.COLOR_BGR2GRAY)
    
    #use adaptive threshold to find hairs and wanted areas
        thresh_adapt = cv.adaptiveThreshold(grayImg, 255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51,1 )
        #enhance all the features using threahold mean and binary with a block size of 51
    
    
        # #use adaptive thr to enhanced edges and hair
        block_size = 9
        adaptive_thresh = threshold_local(grayImg, block_size, offset=10)
        binary_adaptive = grayImg > adaptive_thresh
#5x5 might be too big
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)        
        #get hairs
        # binary_adaptive[thresh_adapt[:,:]==255]=255
        # processImgs.append(np.array(binary_adaptive))
        
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerHairs[:,:]==255]=1
        mask=mask.astype('uint8')
        #img[mask[:,:]==1,:]=255
        img=img.astype('uint8')

        inpainted_img = cv.inpaint(img, mask, inpaintRadius=6, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thresh_adapt[:,:]==0,:]
        
        #replace with the hairs
        img[thresh_adapt[:,:]==0,:]=hair_implants
       
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        #return clean image
        return img
    elif method==5:
        otsu_thresh= threshold_otsu(cv.cvtColor(colourImg,cv.COLOR_BGR2GRAY))
        #get edges using canny's method, otsu thresholding and an upper thresh limit(set by user)
        edges = cv.Canny(colourImg,otsu_thresh,160)
        
        grayImg=grayImg.astype('uint8')
        grayImg[edges[:,:]==255]=0

    #use adaptive threshold to find hairs and wanted areas
        thresh_adapt = cv.adaptiveThreshold(grayImg, 255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51,1 )

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        thickerHairs = cv.dilate(cv.bitwise_not(binary_adaptive.astype('uint8')), kernel)        
        #get hairs
        # binary_adaptive[thresh_adapt[:,:]==255]=255
        # processImgs.append(np.array(binary_adaptive))
        
        mask =np.array( [[0]*imgSize]*imgSize)
        mask[thickerHairs[:,:]==255]=1
        mask=mask.astype('uint8')
        #img[mask[:,:]==1,:]=255
        img=img.astype('uint8')

        inpainted_img = cv.inpaint(img, mask, inpaintRadius=6, flags=cv.INPAINT_TELEA)
        #because thresh adapt is thicker than binary adapt use it to get the hairs 
        hair_implants=inpainted_img[thresh_adapt[:,:]==0,:]
        
        #replace with the hairs
        img[thresh_adapt[:,:]==0,:]=hair_implants
        
        return img
# dataDir=r"C:\Users\yeyit\Documents\FinalProject\ISIC_2019_Training_Input"
# files = os.listdir(dataDir)
# x=0
# while (x<151):
#     index = random.randrange(0, len(files))
#     colourImg=cv.imread(os.path.join(dataDir,files[index]))
#     colourImg=cv.cvtColor(colourImg,cv.COLOR_BGR2RGB)
#     plt.imshow(colourImg)
#     plt.show()
#     x=x+1
# def getBlock():
#     rows, cols = (5, 5)
#     arr =np.array( [[True]*cols]*rows)
#     arr[2,2]=False
#     return arr
# def calcualteDistance(pxl1,pxl2):
#     part1=(pxl2[0]-pxl1[0])**2
#     part2=(pxl2[1]-pxl1[1])**2
#     result=math.sqrt(part1+part2)
#     return result
# def firstHalf():
    
#     return x
    
# def secondHalf():
#     return x

# def getPixelIntensity(toBeReplaced,grayImg):
#     intensity=firstHalf() + secondHalf()
#     return intensity
# def replaceHairs(img):
#     return processedImg