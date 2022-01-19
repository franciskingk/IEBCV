from sre_constants import SUCCESS
import cv2
import pickle
import numpy as np
import cvzone 
import pytesseract


# image
im = cv2.imread('new_ballot-papers/prnairobidagoballot.jpg')
#im = cv2.imread('new_ballot-papers/prnairobidagoballot.jpg')
imgwidth = 860
imgheight = 950
img = cv2.resize(im,(imgwidth,imgheight))


with open('ballotboxposition','rb') as f:
        positionlist= pickle.load(f)
 
# dimension for bounding boxes
width, height = 105, 90


# check the checkboxes for marks
def checkformark(imgpro):
    for pos in positionlist:
        #cv2.rectangle(img,pos)
        x,y = pos
        

        #croping and extracting the image
        imgcrop = imgpro[y:y+height,x:x+width]
        cv2.imshow(str(x*y),imgcrop)   
        count = cv2.countNonZero(imgcrop)
        cvzone.putTextRect(img,str(count),(x,y+height-10),scale=1,thickness=1,offset=0)


        
        # well count the pixels, if they are greater than 1350 then well consider
        #the region marked.
        if count>1350:
            color = (0,255,0)
            thickness = 5



        else:
            color = (0,0,255)
            thickness = 2
        cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),color,thickness)



while True:
    img = cv2.imread('new_ballot-papers/prnairobidagoballot.jpg')
    #img = cv2.imread('new_ballot-papers/prnairobidagoballot.jpg')
    imgwidth = 860
    imgheight = 950
    img = cv2.resize(im,(imgwidth,imgheight))

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
    imgThreshhold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,25,16)
    imgmedialblur = cv2.medianBlur(imgThreshhold,3)
    kernel = np.ones((3,3),np.uint8)
    imgdialate = cv2.dilate(imgmedialblur,kernel, iterations=1)

    checkformark(imgdialate)    

    #for pos in positionlist:
        

    
    cv2.imshow('image',img)
    cv2.imshow('threshhold',imgdialate)

    cv2.waitKey(1)